"""A tiny random Qwen3.5 (same architecture, 16M params) plus the REAL processor, for CPU tests.

Needs network access once, for the processor/config files only (a few MB).
"""

import functools

from mmcqa_exp import settings as S


@functools.lru_cache(maxsize=1)
def build():
    import torch
    from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor
    from mmcqa_exp.backends.qwen import set_pixel_budget

    cfg = AutoConfig.from_pretrained(S.QWEN_REPO, revision=S.QWEN_REVISION)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 4
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 32
    t.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    t.linear_key_head_dim, t.linear_value_head_dim = 16, 16
    t.linear_num_key_heads, t.linear_num_value_heads = 2, 4
    t.mtp_num_hidden_layers = 0
    t.rope_parameters = dict(t.rope_parameters)
    t.rope_parameters["mrope_section"] = [2, 3, 3]
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 2, 64, 128, 4, 64
    torch.manual_seed(0)
    model = AutoModelForMultimodalLM.from_config(cfg)
    processor = AutoProcessor.from_pretrained(S.QWEN_REPO, revision=S.QWEN_REVISION)
    set_pixel_budget(processor)
    return model, processor


def make_images(directory, n=4):
    from PIL import Image
    paths = []
    for i in range(n):
        p = directory / f"img{i}.jpg"
        Image.new("RGB", (300 + 40 * i, 200 + 30 * i), (30 * i, 90, 200 - 20 * i)).save(p)
        paths.append(p)
    return paths
