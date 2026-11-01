import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from local_analysis import judge as Jd
from local_analysis.common import write_jsonl


class FakeJudge:
    def __init__(self):
        self.calls = 0

    def rate(self, question, reference, candidate, track):
        self.calls += 1
        good = reference.split()[0] in candidate
        return {"helpfulness": 8, "relevance": 8, "accuracy": 9 if good else 2, "faithfulness": 8,
                "verdict": "correct" if good else "incorrect", "variety_ok": True, "reason": "x", "judge_model": "fake"}


class JudgeTests(unittest.TestCase):
    def test_cache_dry_run_and_summary(self):
        from local_analysis.common import eval_ids
        ids = eval_ids(3)
        out = Path(tempfile.mkdtemp())
        rows = []
        for s, good in (("sysA", True), ("sysB", False)):
            for t in ("en", "msa", "arz", "ajp"):
                for k, i in enumerate(ids):
                    rows.append({"system": s, "track": t, "id": i, "question": f"q{k}", "reference": f"ref{k} text",
                                 "prediction": f"ref{k} yes" if good else "nope", "mm_f1": 0.8 if good else 0.7})
        write_jsonl(out / "per_item_scores.jsonl", rows)
        args = Namespace(out=out, systems=["sysA", "sysB"], history_systems=[], first_rows=3, model="m", threads=2,
                         env_file=None, dry_run=True)
        dry = Jd.run(args)
        self.assertEqual((dry["jobs"], dry["todo"]), (24, 24))
        args.dry_run = False
        fake = FakeJudge()
        Jd.run(args, judge=fake)
        self.assertEqual(fake.calls, 24)
        again = Jd.run(args, judge=fake)
        self.assertEqual(fake.calls, 24)                              # everything cached: no new calls
        import pandas as pd
        s = pd.read_csv(out / "judge" / "judge_summary.csv").set_index("system")
        self.assertEqual(s.loc["sysA", "correct_pct"], 100.0)
        self.assertEqual(s.loc["sysB", "correct_pct"], 0.0)
        self.assertEqual(again["failed"], 0)

    def test_unknown_system_fails_loudly(self):
        out = Path(tempfile.mkdtemp())
        write_jsonl(out / "per_item_scores.jsonl", [{"system": "a", "track": "en", "id": "x", "question": "q",
                                                      "reference": "r", "prediction": "p", "mm_f1": 0.5}])
        args = Namespace(out=out, systems=["nope"], history_systems=[], first_rows=3, model="m", threads=1,
                         env_file=None, dry_run=True)
        with self.assertRaises(SystemExit):
            Jd.run(args)


if __name__ == "__main__":
    unittest.main()
