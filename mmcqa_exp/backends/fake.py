"""Deterministic stand-in backend for tests of the worker/launcher/merge chain (no model, no GPU)."""

import os
import time


class FakeBackend:
    def __init__(self, adapter=None, four_bit=True):
        self.adapter = adapter

    def load(self):
        return self

    def answer(self, image, prompt, max_new_tokens, do_sample=False, temperature=None, top_p=None):
        if os.environ.get("MMCQA_FAKE_SLEEP"):
            time.sleep(float(os.environ["MMCQA_FAKE_SLEEP"]))
        first = prompt.splitlines()[0]
        text = f"**Answer** to: {first[:40]}" if "FAIL" not in prompt else ""
        return {"raw_answer": text, "new_tokens": 12, "prompt_tokens": 30,
                "hit_max_new_tokens": "CAP" in prompt, "mean_logprob": -1.5}
