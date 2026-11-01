"""Build the Pangea virtualenv and verify it actually works before any GPU time is spent.

Pangea (LLaVA-NeXT) needs transformers 4.37.x while Qwen3.5 needs 5.x, so Pangea gets its own venv that
inherits Kaggle's torch/CUDA/bitsandbytes (--system-site-packages) and overrides only what must differ.
The earlier Pangea run installed these on Kaggle's Python 3.13 without problems; the second recipe is a
fallback that pins tokenizers explicitly.
"""

import subprocess
import sys
from pathlib import Path

from . import settings as S
from .util import package_versions

VERIFY = (
    "import transformers, torch\n"
    "assert transformers.__version__.startswith('4.37'), transformers.__version__\n"
    "from llava.model.builder import load_pretrained_model\n"
    "from llava.mm_utils import tokenizer_image_token, process_images\n"
    "import mmcqa_exp.backends.pangea\n"
    "print('PANGEA_ENV_OK', transformers.__version__, torch.__version__)\n")


def _pip(py, *args):
    cmd = [py, "-m", "pip", "install", "-q", *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True)


def verify(py) -> tuple:
    res = subprocess.run([py, "-c", VERIFY], capture_output=True, text=True, cwd=str(S.REPO_ROOT),
                         env={**__import__("os").environ, "PYTHONPATH": str(S.REPO_ROOT)})
    return res.returncode == 0 and "PANGEA_ENV_OK" in res.stdout, (res.stdout + res.stderr)[-1500:]


RECIPES = {
    "A: requirements + LLaVA-NeXT (--no-deps)": [
        ["-r", str(S.REPO_ROOT / "requirements" / "pangea.txt")],
        ["--no-deps", S.LLAVA_NEXT_GIT]],
    "B: explicit tokenizers/transformers pins": [
        ["--only-binary=:all:", "tokenizers>=0.15,<0.19"],
        ["--no-deps", "--force-reinstall", "transformers==4.37.2"],
        ["-r", str(S.REPO_ROOT / "requirements" / "pangea.txt")],
        ["--no-deps", S.LLAVA_NEXT_GIT]],
}


def setup_pangea_env(env_dir) -> str:
    """Create ``env_dir`` and return its python path. Raises with the pip/import errors if every recipe fails."""
    env_dir = Path(env_dir)
    py = str(env_dir / "bin" / "python")
    if not Path(py).exists():
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(env_dir)], check=True)
    ok, out = verify(py)
    if ok:
        print("[env] Pangea venv already usable:", out.strip(), flush=True)
        return py
    failures = []
    for name, steps in RECIPES.items():
        print(f"[env] trying recipe {name}", flush=True)
        for step in steps:
            res = _pip(py, *step)
            if res.returncode != 0:
                failures.append(f"{name}: pip {' '.join(step)} failed:\n{res.stderr[-800:]}")
                break
        else:
            ok, out = verify(py)
            if ok:
                print("[env]", out.strip(), "| packages:", package_versions(py), flush=True)
                return py
            failures.append(f"{name}: verification failed:\n{out}")
    raise RuntimeError("Could not build a working Pangea environment:\n\n" + "\n\n".join(failures))
