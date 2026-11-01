"""Phase 2 orchestration logic (stage order, dependencies, skipping, merging) with the fake backend."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from mmcqa_exp import jobs as J, launcher, phase2, settings as S


def fake_rows(n=3):
    return {t: [{"id": f"id{i}", "image": f"images/{i}.jpg", "question": f"Question {i} {t}?", "answer": "ref"}
                for i in range(n)] for t in S.TRACKS}


def fake_specs(samples):
    return [J.StageSpec("pangea_p0", "fake", "p0", 16),
            J.StageSpec("qwen_z1", "fake", "p_new", 16),
            J.StageSpec("cascade_qwen2pangea", "fake", "p_cascade", 16, deps=("qwen_z1",)),
            J.StageSpec("fusion_qwen", "fake", "p_fusion", 16, deps=("pangea_p0", "qwen_z1")),
            J.StageSpec("needs_missing", "fake", "p_cascade", 16, deps=("ghost",))]


class Phase2Tests(unittest.TestCase):
    def run_phase2(self, extra=()):
        tmp = Path(tempfile.mkdtemp())
        media = tmp / "media"
        (media / "images").mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (16, 16)).save(media / "images" / f"{i}.jpg")
        launcher.POLL_SECONDS = 0.3
        with mock.patch.object(phase2, "stage_specs", fake_specs), \
             mock.patch.object(phase2.data, "eval_rows", lambda *a, **k: fake_rows()), \
             mock.patch.object(phase2.data, "fetch_images", lambda *a, **k: None), \
             mock.patch.object(phase2.data, "official_repo", lambda w: None), \
             mock.patch.object(launcher, "list_gpu_indices", lambda: [0]), \
             mock.patch.object(phase2, "initial_rates", lambda gpus: {"fake": 10.0, "qwen": 10.0, "pangea": 10.0}), \
             mock.patch.object(S, "WORKERS_PER_GPU", {"qwen": 1, "pangea": 1, "fake": 1}), \
             mock.patch.dict(S.DEFAULT_SEC_PER_ITEM, {"fake": 0.1}), mock.patch.dict(S.SCALING_EFFICIENCY, {"fake": 1.0}):
            phase2.main(["--workdir", str(tmp / "w"), "--media", str(media), "--envs", str(tmp / "envs"),
                         "--budget-hours", "1", *extra])
        return tmp / "w"

    def test_full_chain(self):
        work = self.run_phase2()
        rep = json.loads((work / "logs" / "phase2_completion.json").read_text())["stages"]
        for name in ("pangea_p0", "qwen_z1", "cascade_qwen2pangea", "fusion_qwen"):
            self.assertEqual(rep[name]["state"], "done", rep[name])
            self.assertTrue((work / "predictions" / name / "dev_ajp.jsonl").exists())
        self.assertEqual(rep["needs_missing"]["state"], "skipped")        # a missing dependency skips, never crashes
        casc = [json.loads(l) for l in (work / "predictions" / "cascade_qwen2pangea" / "dev_en.jsonl").read_text().splitlines()]
        self.assertIn("Answer to: Question 0 en?", casc[0]["prompt"])      # hint came from qwen_z1, same track
        self.assertEqual(rep["fusion_qwen"]["status"]["fusion_qwen|msa"]["answered"], 3)

    def test_rerun_single_stage_uses_existing_dependencies(self):
        work = self.run_phase2()
        raw = work / "raw" / "fusion_qwen"
        for f in raw.glob("worker_*.jsonl"):
            f.unlink()
        tmp = work.parent
        with mock.patch.object(phase2, "stage_specs", fake_specs), \
             mock.patch.object(phase2.data, "eval_rows", lambda *a, **k: fake_rows()), \
             mock.patch.object(phase2.data, "fetch_images", lambda *a, **k: None), \
             mock.patch.object(phase2.data, "official_repo", lambda w: None), \
             mock.patch.object(launcher, "list_gpu_indices", lambda: [0]), \
             mock.patch.object(phase2, "initial_rates", lambda gpus: {"fake": 10.0, "qwen": 10.0, "pangea": 10.0}), \
             mock.patch.object(S, "WORKERS_PER_GPU", {"qwen": 1, "pangea": 1, "fake": 1}):
            phase2.main(["--workdir", str(work), "--media", str(tmp / "media"), "--envs", str(tmp / "envs"),
                         "--budget-hours", "1", "--stages", "fusion_qwen"])
        rep = json.loads((work / "logs" / "phase2_completion.json").read_text())["stages"]
        self.assertEqual(rep["fusion_qwen"]["state"], "done")


if __name__ == "__main__":
    unittest.main()
