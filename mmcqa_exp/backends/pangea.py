"""CulturalPangea-7B through LLaVA-NeXT. Ported from models.py (PangeaRunner + the pangea branch of
load_model), which produced our earlier Pangea run on Kaggle. Runs in the Pangea venv
(transformers 4.37.x); never import this module from the main environment.
"""

import contextlib

from .. import settings as S

VISION_MODULES = ("vision_tower", "mm_projector")


def compute_dtype():
    import torch
    native = all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(torch.cuda.device_count()))
    return torch.bfloat16 if native else torch.float16


def quantization_config(dtype):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
        llm_int8_skip_modules=["lm_head", "llm.lm_head", *VISION_MODULES,
                               *(f"model.{name}" for name in VISION_MODULES)])


@contextlib.contextmanager
def inference_context(dtype):
    import torch
    autocast = (torch.autocast("cuda", dtype=dtype) if torch.cuda.is_available()
                and dtype in (torch.float16, torch.bfloat16) else contextlib.nullcontext())
    with torch.inference_mode(), autocast:
        yield


class PangeaBackend:
    def __init__(self, adapter=None, four_bit=True):
        self.four_bit = four_bit
        self.tokenizer = self.model = self.image_processor = self.dtype = None

    def load(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required.")
        try:
            from llava.model.builder import load_pretrained_model
        except ImportError as exc:
            raise RuntimeError("CulturalPangea requires LLaVA-NeXT in the Pangea venv "
                               "(see mmcqa_exp/env_setup.py).") from exc
        self.dtype = compute_dtype()
        print(f"[pangea] loading {S.PANGEA_REPO} dtype={self.dtype} 4bit={self.four_bit}", flush=True)
        try:
            from huggingface_hub import HfApi
            sha = HfApi().model_info(S.PANGEA_REPO).sha
            if sha != S.PANGEA_REVISION:
                print(f"[pangea] WARNING: hub revision {sha} differs from the pinned {S.PANGEA_REVISION}", flush=True)
        except Exception:
            pass
        # Bypass the builder's load_4bit shortcut: it hardcodes FP16 and passes load_in_4bit.
        extra = {"quantization_config": quantization_config(self.dtype)} if self.four_bit else {}
        tokenizer, model, processor, _ = load_pretrained_model(
            S.PANGEA_REPO, None, "CulturalPangea-7B-qwen", multimodal=True,
            torch_dtype=str(self.dtype).split(".")[-1], attn_implementation="sdpa", **extra)
        self.tokenizer, self.model, self.image_processor = tokenizer, model.eval(), processor
        return self

    def answer(self, image, prompt, max_new_tokens, do_sample=False, temperature=None, top_p=None):
        import torch
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token

        text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<image>\n" + prompt +
                "<|im_end|>\n<|im_start|>assistant\n")
        ids = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors="pt").unsqueeze(0).to(self.model.device)
        vision = self.model.get_vision_tower()
        pixels = process_images([image], self.image_processor, self.model.config)
        if isinstance(pixels, list):
            pixels = [value.to(vision.device, dtype=vision.dtype) for value in pixels]
        else:
            pixels = pixels.to(vision.device, dtype=vision.dtype)
        with inference_context(self.dtype):
            output = self.model.generate(ids, images=pixels, image_sizes=[image.size], do_sample=False,
                                         max_new_tokens=max_new_tokens, use_cache=True)
        # LLaVA's generate normally returns new tokens only; trim the input if it is present.
        if output.shape[-1] >= ids.shape[-1] and torch.equal(output[0, :ids.shape[-1]], ids[0]):
            output = output[:, ids.shape[-1]:]
        new = output[0]
        n = int(new.shape[0])
        eos = self.tokenizer.eos_token_id
        hit = bool(n >= max_new_tokens and (n == 0 or int(new[-1]) != eos))
        return {"raw_answer": self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip(),
                "new_tokens": n, "prompt_tokens": int(ids.shape[-1]), "hit_max_new_tokens": hit,
                "mean_logprob": None}      # not collected for LLaVA: generate() scores are not exposed reliably
