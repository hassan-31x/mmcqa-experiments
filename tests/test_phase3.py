"""Phase 3 (quantized Qwen3.5-9B zero-shot) orchestration with the fake backend, and the backend registry."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from mmcqa_exp import launcher, phase3, settings as S
from mmcqa_exp.backends import get_backend


class Phase3Tests(unittest.TestCase):
    def test_registry_pins_the_9b(self):
        cls = get_backend("qwen9b")
        self.assertEqual((cls.repo, cls.revision), (S.QWEN9B_REPO, S.QWEN9B_REVISION))
        self.assertNotEqual(cls.repo, get_backend("qwen").repo)

    def test_stage_runs_and_merges(self):
        tmp = Path(tempfile.mkdtemp())
        media = tmp / "media"
        (media / "images").mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (16, 16)).save(media / "images" / f"{i}.jpg")
        rows = {t: [{"id": f"id{i}", "image": f"images/{i}.jpg", "question": f"Question {i} {t}?", "answer": "ref"}
                    for i in range(3)] for t in S.TRACKS}
        launcher.POLL_SECONDS = 0.3
        with mock.patch.object(phase3.data, "eval_rows", lambda *a, **k: rows), \
             mock.patch.object(phase3.data, "fetch_images", lambda *a, **k: None), \
             mock.patch.object(phase3.data, "official_repo", lambda w: None), \
             mock.patch.object(launcher, "list_gpu_indices", lambda: [0]):
            phase3.main(["--workdir", str(tmp / "w"), "--media", str(media), "--rows", "3", "--backend", "fake"])
        done = json.loads((tmp / "w" / "logs" / "phase3_completion.json").read_text())
        self.assertEqual(done["run"]["missing"], 0)
        self.assertEqual(sum(v["answered"] for v in done["status"].values()), 12)
        self.assertTrue((tmp / "w" / "predictions" / "qwen9b_z1" / "dev_arz.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
