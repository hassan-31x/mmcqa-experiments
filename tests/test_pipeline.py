"""Worker -> launcher -> merge chain with real subprocesses and the fake backend (no GPU, no model)."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mmcqa_exp import jobs as J, launcher, merge, settings as S
from mmcqa_exp.util import read_jsonl, write_jsonl


def fake_rows(n=3, ids=None):
    ids = ids or [f"id{i}" for i in range(n)]
    return {t: [{"id": rid, "image": f"images/{i % 3}.jpg", "question": f"Question {i} {t}?" + (" FAIL" if i == 1 else ""),
                 "answer": "ref"} for i, rid in enumerate(ids)] for t in S.TRACKS}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.media = self.tmp / "media"
        (self.media / "images").mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (32, 32)).save(self.media / "images" / f"{i}.jpg")
        launcher.POLL_SECONDS = 0.3

    def test_plan_run_merge_resume(self):
        work = self.tmp / "work"
        spec = J.StageSpec("fake_a", "fake", "p_new", 32)
        jobs = J.build_plan(spec, work, fake_rows())
        self.assertEqual(len(jobs), 12)
        res = launcher.run_stage(work, "fake_a", __import__("sys").executable, [0], 2, 0, self.media)
        self.assertEqual(res["missing"], 0)
        st = merge.merge_stage(work, "fake_a")
        for track in S.TRACKS:
            s = st[f"fake_a|{track}"]
            self.assertEqual((s["planned"], s["answered"], s["errors"]), (3, 3, 0))
            self.assertEqual(s["fallbacks"], 1)                        # the "FAIL" prompt returns "" -> fallback
        rows = read_jsonl(work / "predictions" / "fake_a" / "dev_en.jsonl")
        self.assertTrue(all(r["answer"].strip() for r in rows))
        self.assertNotIn("**", rows[0]["answer"])                      # markdown stripped
        sub = read_jsonl(work / "submission" / "fake_a" / "dev" / "prediction_qa_ajp.jsonl")
        self.assertEqual({tuple(sorted(r)) for r in sub}, {("answer", "id", "lang")})
        # a second run answers nothing new
        again = launcher.run_stage(work, "fake_a", __import__("sys").executable, [0], 2, 0, self.media)
        self.assertEqual(again["missing"], 0)

    def test_killed_worker_line_and_resume(self):
        work = self.tmp / "work2"
        spec = J.StageSpec("fake_b", "fake", "p_new", 32)
        J.build_plan(spec, work, fake_rows())
        raw = work / "raw" / "fake_b"
        raw.mkdir(parents=True)
        (raw / "worker_0.jsonl").write_text('{"key": "fake_b|en|id0", "error": null, "answer": "x"}\n{"key": "fake_b|en|id1", "err')
        self.assertEqual(len(launcher.missing_keys(work, "fake_b")), 11)   # truncated line ignored
        res = launcher.run_stage(work, "fake_b", __import__("sys").executable, [0], 1, 0, self.media)
        self.assertEqual(res["missing"], 0)

    def test_cascade_and_fusion_prompts(self):
        work = self.tmp / "work3"
        from mmcqa_exp import prompts
        pool = [f"id{i}" for i in range(40)]
        ids = [next(i for i in pool if prompts.fusion_order(i) == 0), next(i for i in pool if prompts.fusion_order(i) == 1), "other"]
        rows = fake_rows(ids=ids)
        for name in ("pangea_p0", "qwen_z1"):
            spec = J.StageSpec(name, "fake", "p_new", 32)
            J.build_plan(spec, work, rows)
            launcher.run_stage(work, name, __import__("sys").executable, [0], 1, 0, self.media)
            merge.merge_stage(work, name)
        casc = J.build_plan(J.StageSpec("c", "fake", "p_cascade", 32, deps=("qwen_z1",)), work, rows)
        self.assertIn("Answer to: Question 0 en?", casc[0]["prompt"])   # same-track hint from qwen_z1
        fus = J.build_plan(J.StageSpec("f", "fake", "p_fusion", 32, deps=("pangea_p0", "qwen_z1")), work, rows)
        self.assertEqual({j["meta"]["fusion_a"] for j in fus}, {"pangea_p0", "qwen_z1"})
        self.assertTrue(all("Candidate answer A:" in j["prompt"] for j in fus))
        with self.assertRaises(RuntimeError):
            J.build_plan(J.StageSpec("x", "fake", "p_cascade", 32, deps=("missing_system",)), work, rows)

    def test_errors_fall_back_and_retry(self):
        work = self.tmp / "work4"
        spec = J.StageSpec("fake_c", "fake", "p_new", 32)
        J.build_plan(spec, work, fake_rows(1))
        raw = work / "raw" / "fake_c"
        raw.mkdir(parents=True)
        write_jsonl(raw / "worker_0.jsonl", [{"key": "fake_c|en|id0", "stage": "fake_c", "system": "fake_c", "split": "dev",
            "track": "en", "id": "id0", "raw_answer": "", "answer": "echo", "fallback_used": True, "error": "OutOfMemoryError: x",
            "latency_seconds": 1, "new_tokens": None, "prompt_tokens": None, "hit_max_new_tokens": None,
            "mean_logprob": None, "peak_mem_gb": None, "worker": 0, "finished_unix": 0}])
        self.assertIn("fake_c|en|id0", launcher.missing_keys(work, "fake_c"))   # an errored answer is retried
        launcher.run_stage(work, "fake_c", __import__("sys").executable, [0], 1, 0, self.media)
        best = merge.latest_answers(raw)["fake_c|en|id0"]
        self.assertIsNone(best["error"])                                   # the error-free retry wins


if __name__ == "__main__":
    unittest.main()
