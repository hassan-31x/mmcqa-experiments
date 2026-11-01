"""Install the Pangea stack (transformers 4.37 + LLaVA-NeXT) into its own directory and verify it works.

Pangea needs transformers 4.37.x while Qwen3.5 needs 5.x. Instead of a virtualenv (Kaggle's system Python has
no ensurepip) the Pangea packages go into a ``pip --target`` directory that Pangea workers put first on
PYTHONPATH, so it shadows the main environment's transformers/tokenizers/huggingface_hub for those workers only.
Everything is installed with --no-deps so pip never touches torch or other heavy shared packages.
"""

import os
import subprocess
import sys
from pathlib import Path

from . import settings as S

VERIFY = (
    "import transformers, tokenizers, huggingface_hub, torch, accelerate\n"
    "from accelerate import dispatch_model\n"
    "assert accelerate.__version__.startswith('0.27'), accelerate.__version__\n"
    "assert transformers.__version__.startswith('4.37'), transformers.__version__\n"
    "from llava.model.builder import load_pretrained_model\n"
    "from llava.mm_utils import tokenizer_image_token, process_images\n"
    "import mmcqa_exp.backends.pangea\n"
    "print('PANGEA_ENV_OK', 'transformers', transformers.__version__, 'tokenizers', tokenizers.__version__,\n"
    "      'hub', huggingface_hub.__version__, 'accelerate', accelerate.__version__, 'torch', torch.__version__)\n")

# accelerate 0.27.x is the era-matched version: newer ones call .to() on 4-bit models, which transformers 4.37 forbids
SHARED = ["accelerate==0.27.2", "huggingface_hub<1.0", "shortuuid", "ftfy", "wcwidth", "open_clip_torch", "timm", "einops-exts"]
RECIPES = {
    "A: transformers 4.37.2 + newest compatible tokenizers wheel": [
        ["transformers==4.37.2", "--only-binary=:all:", "tokenizers>=0.15,<0.19", *SHARED],
        [S.LLAVA_NEXT_GIT]],
    "B: transformers 4.37.2 + tokenizers 0.15.2 (any build)": [
        ["transformers==4.37.2", "tokenizers==0.15.2", *SHARED],
        [S.LLAVA_NEXT_GIT]],
}


def pangea_env(target) -> dict:
    """Environment variables for running a Pangea worker."""
    return {"PYTHONPATH": f"{target}{os.pathsep}{S.REPO_ROOT}"}


def verify(python, target) -> tuple:
    env = {**os.environ, **pangea_env(target), "TOKENIZERS_PARALLELISM": "false"}
    res = subprocess.run([python, "-c", VERIFY], capture_output=True, text=True, cwd=str(S.REPO_ROOT), env=env)
    return res.returncode == 0 and "PANGEA_ENV_OK" in res.stdout, (res.stdout + res.stderr)[-2500:]


def setup_pangea_env(target) -> str:
    """Install into ``target``; return the PYTHONPATH-ready target directory. Raises with every pip/import error."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    ok, out = verify(sys.executable, target)
    if ok:
        print("[env] Pangea packages already usable:", out.strip(), flush=True)
        return str(target)
    failures = []
    for name, steps in RECIPES.items():
        print(f"[env] trying recipe {name}", flush=True)
        for step in steps:
            cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--upgrade", "--target", str(target), *step]
            print("+", " ".join(cmd), flush=True)
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                failures.append(f"{name}: pip failed for {step[:3]}:\n{(res.stdout + res.stderr)[-1500:]}")
                break
        else:
            ok, out = verify(sys.executable, target)
            if ok:
                print("[env]", out.strip(), flush=True)
                return str(target)
            failures.append(f"{name}: verification failed:\n{out}")
    raise RuntimeError("Could not build a working Pangea environment:\n\n" + "\n\n".join(failures))
