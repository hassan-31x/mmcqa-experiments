"""Small helpers shared by every module. Standard library only (workers run in two environments)."""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _json_default(o):
    """numpy/pandas scalars -> Python scalars; anything else -> str."""
    return o.item() if hasattr(o, "item") else str(o)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path) -> list:
    """Read a JSONL file, skipping a final line cut short by a killed process."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_jsonl_many(folder, pattern="worker_*.jsonl") -> list:
    return [row for path in sorted(Path(folder).glob(pattern)) for row in read_jsonl(path)]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def gpu_names() -> list:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.split(":", 1)[1].split("(")[0].strip() for line in out.splitlines() if line.startswith("GPU")]


def package_versions(python_exe=None, names=("torch", "transformers", "peft", "bitsandbytes", "accelerate",
                                              "huggingface_hub", "tokenizers", "numpy", "pillow")) -> dict:
    """Installed versions as seen by ``python_exe`` (default: the running interpreter)."""
    code = ("import importlib.metadata as m, json, sys\nout={}\nfor n in sys.argv[1:]:\n"
            "    try: out[n]=m.version(n)\n    except Exception: out[n]=None\nprint(json.dumps(out))")
    try:
        res = subprocess.run([python_exe or sys.executable, "-c", code, *names],
                             capture_output=True, text=True, check=True)
        return json.loads(res.stdout.strip().splitlines()[-1])
    except Exception as exc:                                  # a broken env must not stop the run
        return {"error": f"{type(exc).__name__}: {exc}"}


def environment(extra_pythons=None) -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "gpus": gpu_names(),
            "packages": package_versions()}
    for label, exe in (extra_pythons or {}).items():
        info[f"packages_{label}"] = package_versions(exe)
    return info


def free_disk_gb(path) -> float:
    import shutil
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 2**30


def hours_since(t0: float) -> float:
    return (time.time() - t0) / 3600


def sh(cmd, check=True, env=None, cwd=None):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check, env=env, cwd=cwd)


def words(text) -> int:
    return len(str(text).split())


def cap_words(text, n) -> str:
    parts = str(text).split()
    return " ".join(parts[:n]) if len(parts) > n else str(text).strip()


def env_with(**kw):
    return dict(os.environ, **{k: str(v) for k, v in kw.items()})
