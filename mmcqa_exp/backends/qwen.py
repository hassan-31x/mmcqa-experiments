"""Qwen3.5-4B (NF4, optional LoRA adapter). Used for zero-shot stages, SFT inference and, through
``load_qwen``/``set_pixel_budget``, for SFT training, so all three see identical images and prompts."""

import contextlib

from .. import settings as S

# Matches the loader of our earlier Qwen3.5 runs (models.py: VISION_MODULES + lm_head skips).
SKIP_MODULES = ["lm_head", "llm.lm_head", "visual", "model.visual"]


def compute_dtype():
    """BF16 only if every visible GPU supports it natively (T4 does not), else FP16."""
    import torch
    if not torch.cuda.is_available():
        return torch.float32
    native = all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(torch.cuda.device_count()))
    return torch.bfloat16 if native else torch.float16


def set_pixel_budget(processor, max_pixels=S.QWEN_MAX_PIXELS, min_pixels=S.QWEN_MIN_PIXELS):
    """Cap image resolution. Called by inference AND training so the visual token count matches."""
    ip = processor.image_processor
    size = {"longest_edge": int(max_pixels), "shortest_edge": int(min_pixels)}
    try:
        from transformers.image_utils import SizeDict
        ip.size = SizeDict(**size)
    except Exception:
        ip.size = size
    return processor


def image_token_count(processor, image) -> int:
    """Number of visual tokens the processor produces for ``image``."""
    ip = processor.image_processor
    grid = ip(images=[image], return_tensors="pt")["image_grid_thw"][0]
    merge = getattr(ip, "merge_size", 2)
    return int(grid.prod().item() // (merge * merge))


def assert_pixel_budget(processor, max_pixels=S.QWEN_MAX_PIXELS) -> int:
    """Fail loudly if the pixel cap did not take effect (it would silently change memory and results)."""
    from PIL import Image
    ip = processor.image_processor
    unit = getattr(ip, "patch_size", 16) * getattr(ip, "merge_size", 2)
    tokens = image_token_count(processor, Image.new("RGB", (1400, 1000)))
    limit = max_pixels // (unit * unit) + 4
    if tokens > limit:
        raise RuntimeError(f"Pixel budget not applied: a 1400x1000 image gives {tokens} visual tokens, "
                           f"expected at most {limit}. Check processor.image_processor.size.")
    return tokens


def bnb_config(dtype):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=SKIP_MODULES)


def load_qwen(four_bit=True):
    """Load base model + processor on the single visible GPU. Returns ``(model, processor, dtype)``."""
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required (Kaggle: Settings -> Accelerator -> GPU T4 x2).")
    dtype = compute_dtype()
    print(f"[qwen] loading {S.QWEN_REPO}@{S.QWEN_REVISION[:8]} dtype={dtype} 4bit={four_bit}", flush=True)
    kwargs = {"revision": S.QWEN_REVISION, "device_map": {"": 0}, "dtype": dtype}
    if four_bit:
        kwargs["quantization_config"] = bnb_config(dtype)
    model = AutoModelForMultimodalLM.from_pretrained(S.QWEN_REPO, **kwargs)
    processor = AutoProcessor.from_pretrained(S.QWEN_REPO, revision=S.QWEN_REVISION)
    set_pixel_budget(processor)
    tokens = assert_pixel_budget(processor)
    print(f"[qwen] pixel budget {S.QWEN_MAX_PIXELS} -> {tokens} visual tokens for a 1400x1000 image", flush=True)
    return model, processor, dtype


class FiniteLogitsCheck:
    """Stop numerical failures before greedy argmax turns NaNs into token 0 (ported from models.py)."""

    def __call__(self, input_ids, scores):
        import torch
        if (torch.isnan(scores).any() or torch.isposinf(scores).any()
                or not torch.isfinite(scores).any(dim=-1).all()):
            raise RuntimeError("Model produced non-finite generation scores (NaN/Inf).")
        return scores


@contextlib.contextmanager
def inference_context(dtype):
    import torch
    autocast = (torch.autocast("cuda", dtype=dtype) if torch.cuda.is_available()
                and dtype in (torch.float16, torch.bfloat16) else contextlib.nullcontext())
    with torch.inference_mode(), autocast:
        yield


def _eos_ids(model, processor) -> set:
    ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if ids is None:
        ids = getattr(processor.tokenizer, "eos_token_id", None)
    ids = ids if isinstance(ids, (list, tuple)) else [ids]
    return {int(i) for i in ids if i is not None}


class QwenBackend:
    def __init__(self, adapter=None, four_bit=True):
        self.adapter, self.four_bit = adapter, four_bit
        self.model = self.processor = self.dtype = None

    @classmethod
    def from_loaded(cls, model, processor, dtype):
        """Wrap an already built model (used by the CPU tests)."""
        self = cls()
        self.model, self.processor, self.dtype = model, processor, dtype
        return self

    def load(self):
        self.model, self.processor, self.dtype = load_qwen(self.four_bit)
        if self.adapter:
            from peft import PeftModel
            print(f"[qwen] attaching LoRA adapter {self.adapter}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, self.adapter, is_trainable=False)
        self.model.eval()
        return self

    def answer(self, image, prompt, max_new_tokens, do_sample=False, temperature=None, top_p=None):
        import torch
        from transformers import LogitsProcessorList
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False)
        # BatchFeature.to casts floating tensors only; ids/masks/grids keep their integer dtype.
        inputs = inputs.to(self.model.device, dtype=self.dtype)
        length = inputs["input_ids"].shape[-1]
        gen = {"max_new_tokens": max_new_tokens, "do_sample": bool(do_sample), "output_scores": True,
               "return_dict_in_generate": True, "logits_processor": LogitsProcessorList([FiniteLogitsCheck()])}
        if do_sample:
            if temperature:
                gen["temperature"] = float(temperature)
            if top_p:
                gen["top_p"] = float(top_p)
        with inference_context(self.dtype):
            out = self.model.generate(**inputs, **gen)
        tokens = out.sequences[0][length:]
        mean_logprob = None
        try:
            ts = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
            ts = ts[torch.isfinite(ts)]
            mean_logprob = round(float(ts.float().mean()), 4) if ts.numel() else None
        except Exception:
            pass
        text = self.processor.decode(tokens, skip_special_tokens=True).strip()
        n = int(tokens.shape[0])
        last = int(tokens[-1]) if n else None
        return {"raw_answer": text, "new_tokens": n, "prompt_tokens": int(length),
                "hit_max_new_tokens": bool(n >= max_new_tokens and last not in _eos_ids(self.model, self.processor)),
                "mean_logprob": mean_logprob}
