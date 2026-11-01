"""CPU-only probe: can the Pangea packages be installed and imported? (no GPU quota used)"""
import subprocess, sys, os
def sh(c): print("+", c, flush=True); subprocess.run(c, shell=True)
sh("rm -rf /kaggle/working/mmcqa-experiments && git clone -q https://github.com/hassan-31x/mmcqa-experiments.git /kaggle/working/mmcqa-experiments")
sh("pip install -q -r /kaggle/working/mmcqa-experiments/requirements/qwen.txt 2>&1 | tail -2")
sh(f"which -a python python3; {sys.executable} -V; python -c 'import sys;print(sys.executable)'")
os.chdir("/kaggle/working/mmcqa-experiments"); sys.path.insert(0, ".")
from mmcqa_exp.env_setup import setup_pangea_env
try:
    print("RESULT:", setup_pangea_env("/kaggle/temp/pangea_site"))
except Exception as e:
    print("FAILED:\n", e)
