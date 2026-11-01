"""SFT path on a tiny random Qwen3.5 with the REAL processor and chat template (CPU).

These tests answer the question that matters most: does training learn exactly the answer tokens,
with the same prefix the inference path produces?
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mmcqa_exp import prompts, settings as S
from mmcqa_exp.backends.qwen import QwenBackend, assert_pixel_budget, image_token_count, set_pixel_budget
from mmcqa_exp.sft import train as T
from mmcqa_exp.sft.collate import SFTCollator, prompt_text_for
from mmcqa_exp.sft.configs import SFTConfig
from mmcqa_exp.sft.dataset import build_examples

from . import tiny_qwen

ANSWERS = [
    "The Burj Khalifa, the tallest building in the world, is in Dubai.",
    "برج خليفة هو أطول مبنى في العالم ويقع في دبي.",
    "1. A dish called kunafa; it is sweet - \"very\" sweet! (see 2024).",
    "شكشوكة: طبق مصري. يُقدَّم ساخنًا مع الخبز؟",
    "Short.",
]


def make_examples(tmp, n=12):
    paths = tiny_qwen.make_images(tmp, 4)
    ex = []
    for i in range(n):
        track = S.TRACKS[i % 4]
        ex.append({"id": f"id{i}", "track": track, "image": str(paths[i % 4]),
                   "prompt": prompts.p_new(f"Question {i}?", track), "answer": ANSWERS[i % len(ANSWERS)]})
    return ex


class TinyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.model, cls.processor = tiny_qwen.build()
        except Exception as exc:                        # offline without a cached processor
            raise unittest.SkipTest(f"tiny Qwen3.5 could not be built: {exc}")
        cls.tmp = Path(tempfile.mkdtemp())

    def test_pixel_budget_takes_effect(self):
        tokens = assert_pixel_budget(self.processor)
        self.assertLessEqual(tokens, S.QWEN_MAX_PIXELS // (32 * 32) + 4)

    def test_labels_are_exactly_the_answer(self):
        collator = SFTCollator(self.processor)
        examples = make_examples(self.tmp, 10)
        batch = collator(examples[:5])
        tok = self.processor.tokenizer
        for i, ex in enumerate(examples[:5]):
            lab = batch["labels"][i]
            self.assertEqual(tok.decode(lab[lab != -100]), ex["answer"].strip() + "<|im_end|>")
            ids = batch["input_ids"][i].tolist()
            start = int((lab != -100).nonzero()[0])
            # the prefix up to the answer is what inference sees: template with generation prompt
            pre = self.processor(text=[prompt_text_for(self.processor, ex["prompt"])],
                                 images=[__import__("PIL.Image", fromlist=["Image"]).open(ex["image"]).convert("RGB")],
                                 return_tensors="pt")["input_ids"][0].tolist()
            self.assertEqual(ids[:start], pre)
        padded = batch["attention_mask"] == 0
        self.assertTrue((batch["labels"][padded] == -100).all())

    def test_training_updates_only_lora_and_loss_falls(self):
        import copy
        model = copy.deepcopy(self.model)
        cfg = SFTConfig("T", lr=3e-3, per_device_bs=2, grad_accum=2, epochs=20, max_steps=40, eval_every=20,
                        num_workers=0, min_lora_dim=16, warn_below_steps=0)
        work = Path(tempfile.mkdtemp())
        (work / "logs").mkdir()
        base_vis = {n: p.clone() for n, p in model.named_parameters() if "visual" in n}
        model = T.prepare_lora(model, cfg, work / "logs")
        targets = (work / "logs" / "sft_T_modules.txt").read_text()
        self.assertNotIn("visual", targets.split("target modules:")[1])
        examples = make_examples(self.tmp, 8)
        summary = T.run_training(model, self.processor, cfg, examples, examples[:4], work, 0,
                                 torch.device("cpu"), torch.float32)
        self.assertEqual(summary["steps_done"], 40)
        self.assertTrue((work / "adapters" / "T" / "adapter_config.json").exists())
        rows = [r for r in (work / "logs" / "sft_T_history.csv").read_text().splitlines()[1:] if r]
        losses = [float(r.split(",")[1]) for r in rows if r.split(",")[1]]
        self.assertLess(losses[-1], losses[0])                 # it learns
        self.assertLess(summary["final_eval_loss"], 12.0)   # ln(vocab)=12.4 at chance
        for n, p in model.named_parameters():                   # frozen things did not move
            if "visual" in n:
                key = n.replace("base_model.model.", "")
                self.assertTrue(torch.equal(p.detach(), base_vis[key]), n)

    def test_backend_answer_and_adapter_reload(self):
        import copy
        from peft import PeftModel
        work = Path(tempfile.mkdtemp())
        (work / "logs").mkdir()
        cfg = SFTConfig("R", lr=1e-3, per_device_bs=2, grad_accum=1, max_steps=3, eval_every=2, num_workers=0,
                        min_lora_dim=16, warn_below_steps=0)
        model = T.prepare_lora(copy.deepcopy(self.model), cfg, work / "logs")
        examples = make_examples(self.tmp, 4)
        T.run_training(model, self.processor, cfg, examples, examples[:2], work, 0, torch.device("cpu"), torch.float32)
        base = copy.deepcopy(self.model)
        backend = QwenBackend.from_loaded(base, self.processor, torch.float32)
        res = backend.answer(__import__("PIL.Image", fromlist=["Image"]).new("RGB", (300, 200)), "Q?", 6)
        self.assertEqual(set(res), {"raw_answer", "new_tokens", "prompt_tokens", "hit_max_new_tokens", "mean_logprob"})
        self.assertLessEqual(res["new_tokens"], 6)
        self.assertIsNotNone(res["mean_logprob"])
        reloaded = PeftModel.from_pretrained(copy.deepcopy(self.model), str(work / "adapters" / "R"))
        out = QwenBackend.from_loaded(reloaded.eval(), self.processor, torch.float32).answer(
            __import__("PIL.Image", fromlist=["Image"]).new("RGB", (300, 200)), "Q?", 4)
        self.assertGreaterEqual(out["new_tokens"], 1)

    def test_build_examples_skips_empty_answers(self):
        rows = {"en": [{"id": "a", "image": "images/a.jpg", "question": "Q", "answer": " "},
                       {"id": "b", "image": "images/b.jpg", "question": "Q", "answer": "A"}]}
        ex, skipped = build_examples(rows, "/m", shuffle_seed=1)
        self.assertEqual((len(ex), skipped), (1, 1))
        self.assertEqual(ex[0]["image"], "/m/images/b.jpg")


if __name__ == "__main__":
    unittest.main()
