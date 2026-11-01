"""Stage specifications and the job list (plan) each stage generates.

A job carries the fully rendered prompt, so a worker never needs prompt logic and the exact text
sent to the model is stored next to its answer.
"""

from dataclasses import dataclass, field
from pathlib import Path

from . import prompts, settings as S
from .util import read_jsonl, write_jsonl


@dataclass
class StageSpec:
    name: str                       # also the system name in every output file
    backend: str                    # "qwen" | "pangea"
    prompt: str                     # "p0" | "p_new" | "p_cascade" | "p_fusion"
    max_new_tokens: int
    split: str = "dev"
    deps: tuple = ()                # systems whose merged predictions this stage's prompts quote
    adapter: str = None             # LoRA adapter directory, relative to the workdir
    do_sample: bool = False
    temperature: float = None
    top_p: float = None
    seed: int = None
    note: str = ""

    def describe(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def stage_paths(work, stage: str) -> dict:
    work = Path(work)
    return {"plan": work / "plan" / f"{stage}.jsonl", "raw": work / "raw" / stage,
            "pred": work / "predictions" / stage, "submission": work / "submission" / stage}


def load_dep_answers(work, system: str, split: str) -> dict:
    """``{(track, id): cleaned answer}`` from a merged earlier stage."""
    out = {}
    for track in S.TRACKS:
        path = Path(work) / "predictions" / system / f"{split}_{track}.jsonl"
        for row in read_jsonl(path):
            out[(track, row["id"])] = row["answer"]
    if not out:
        raise RuntimeError(f"No merged predictions found for dependency '{system}' ({split}).")
    return out


def render_prompt(spec: StageSpec, row: dict, track: str, deps: dict):
    """Return ``(prompt, meta)`` for one row."""
    q, rid = row["question"], row["id"]
    if spec.prompt == "p0":
        return prompts.p0(q, track), {}
    if spec.prompt == "p_new":
        return prompts.p_new(q, track), {}
    if spec.prompt == "p_cascade":
        src = spec.deps[0]
        return prompts.p_cascade(q, track, deps[src][(track, rid)]), {"hint_from": src}
    if spec.prompt == "p_fusion":
        first, second = spec.deps[0], spec.deps[1]
        if prompts.fusion_order(rid) == 1:
            first, second = second, first
        a, b = deps[first][(track, rid)], deps[second][(track, rid)]
        return prompts.p_fusion(q, track, a, b), {"fusion_a": first, "fusion_b": second}
    raise ValueError(f"Unknown prompt kind: {spec.prompt}")


def build_plan(spec: StageSpec, work, rows_by_track: dict) -> list:
    """Write ``plan/<stage>.jsonl`` and return the jobs."""
    deps = {d: load_dep_answers(work, d, spec.split) for d in spec.deps}
    jobs = []
    for track in S.TRACKS:
        for row in rows_by_track.get(track, []):
            prompt, meta = render_prompt(spec, row, track, deps)
            jobs.append({
                "key": f"{spec.name}|{track}|{row['id']}", "stage": spec.name, "system": spec.name,
                "split": spec.split, "track": track, "id": row["id"], "image": row["image"],
                "question": row["question"], "prompt": prompt, "backend": spec.backend,
                "adapter": spec.adapter, "max_new_tokens": spec.max_new_tokens,
                "do_sample": spec.do_sample, "temperature": spec.temperature, "top_p": spec.top_p,
                "seed": spec.seed, "meta": meta})
    write_jsonl(stage_paths(work, spec.name)["plan"], jobs)
    return jobs
