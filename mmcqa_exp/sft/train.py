"""One QLoRA training process (one GPU, one config).

    CUDA_VISIBLE_DEVICES=0 python -m mmcqa_exp.sft.train --config S1 --workdir W --media M --deadline T

Hand-written loop (no HF Trainer) so that the loss masking, fp16 scaling, OOM handling and the time
budget are explicit. Outputs, under the workdir:
    adapters/<cfg>/                      best LoRA adapter (lowest eval loss)
    logs/sft_<cfg>_history.csv           step, loss, eval_loss, lr, grad_norm, elapsed, memory
    logs/sft_<cfg>_sizing.json           measured sec/step and the chosen number of steps
    logs/sft_<cfg>_modules.txt           LoRA target modules and trainable parameter count
    logs/sft_<cfg>_summary.json          result (or logs/sft_<cfg>_failed.json on failure)
"""

import argparse
import contextlib
import csv
import json
import math
import sys
import time
from pathlib import Path

from .. import data, settings as S
from ..util import dump_json
from .collate import SFTCollator
from .configs import CONFIGS, SFTConfig
from .dataset import ListDataset, build_examples

EARLY_STOP_MARGIN_S = 8 * 60       # stop training this long before the deadline (final eval + save)
SIZING_MARGIN_S = 15 * 60          # reserve at sizing time
STEP_OVERHEAD = 1.08               # eval/save overhead on top of measured step time
MAX_OOM_SKIPS = 5
MAX_NONFINITE_MICROBATCHES = 8


# ----------------------------------------------------------------------------- helpers
def to_device(batch: dict, device, dtype) -> dict:
    out = {}
    for k, v in batch.items():
        if k == "labels":
            continue
        out[k] = v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
    return out


def lm_loss(logits, labels):
    """Sum of token cross-entropy over answer positions, and the token count."""
    import torch.nn.functional as F
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
    mask = shift_labels != -100
    selected = shift_logits[mask]
    return F.cross_entropy(selected.float(), shift_labels[mask], reduction="sum"), int(mask.sum())


def make_loader(examples, collator, batch_size, shuffle, seed, num_workers):
    import torch
    from torch.utils.data import DataLoader
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ListDataset(examples), batch_size=batch_size, shuffle=shuffle, collate_fn=collator,
                      num_workers=num_workers, generator=gen, drop_last=shuffle,
                      persistent_workers=False)


def batches_forever(examples, collator, batch_size, seed, num_workers):
    epoch = 0
    while True:
        for batch in make_loader(examples, collator, batch_size, True, seed + epoch, num_workers):
            yield batch
        epoch += 1


def lora_targets(model, min_dim: int) -> list:
    """Full names of the language model's Linear layers (4-bit or plain), excluding vision/head/embeddings."""
    import torch.nn as nn
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        low = name.lower()
        if "visual" in low or "lm_head" in low or "embed" in low or "mtp" in low:
            continue
        if min(module.in_features, module.out_features) < min_dim:
            continue
        names.append(name)
    return names


def prepare_lora(model, cfg: SFTConfig, log_dir: Path):
    import torch
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    targets = lora_targets(model, cfg.min_lora_dim)
    if len(targets) < 8:
        raise RuntimeError(f"Only {len(targets)} LoRA target modules found: {targets[:5]}")
    lora = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lora)
    for p in model.parameters():                    # fp16 GradScaler needs fp32 trainable weights
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    (log_dir / f"sft_{cfg.name}_modules.txt").write_text(
        f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)\n"
        f"{len(targets)} target modules:\n" + "\n".join(targets) + "\n")
    print(f"[sft {cfg.name}] LoRA on {len(targets)} modules, {trainable / 1e6:.1f}M trainable parameters", flush=True)
    return model


def autocast_for(dtype):
    import torch
    if torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16):
        return lambda: torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext


def make_optimizer(params, cfg: SFTConfig):
    import torch
    try:
        import bitsandbytes as bnb
        print(f"[sft {cfg.name}] optimizer: bitsandbytes PagedAdamW8bit", flush=True)
        return bnb.optim.PagedAdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    except Exception as exc:
        print(f"[sft {cfg.name}] bitsandbytes optimizer unavailable ({exc}); using torch AdamW", flush=True)
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def forward_loss(model, batch, device, dtype, ctx):
    labels = batch["labels"].to(device)
    with ctx():
        out = model(**to_device(batch, device, dtype), use_cache=False)
        loss_sum, n_tokens = lm_loss(out.logits, labels)
    return loss_sum, n_tokens


def evaluate(model, collator, examples, batch_size, device, dtype, ctx) -> float:
    import torch
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in make_loader(examples, collator, batch_size, False, 0, 0):
            loss_sum, n = forward_loss(model, batch, device, dtype, ctx)
            total += float(loss_sum)
            count += n
    model.train()
    return total / max(count, 1)


def probe(model, collator, examples, cfg: SFTConfig, device, dtype, ctx, steps=3):
    """Measure seconds per optimizer step and peak memory, stepping down the micro-batch on OOM."""
    import torch
    options, bs = [], cfg.per_device_bs
    while bs >= 1:
        options.append((bs, cfg.effective_batch // bs))
        bs //= 2
    cuda = device.type == "cuda"
    total_mem = torch.cuda.get_device_properties(device).total_memory if cuda else None
    for bs, accum in options:
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            gen = batches_forever(examples, collator, bs, 0, cfg.num_workers)
            times = []
            model.train()
            for s in range(steps):
                t0 = time.time()
                for _ in range(accum):
                    loss_sum, n = forward_loss(model, next(gen), device, dtype, ctx)
                    (loss_sum / max(n, 1) / accum).backward()
                if cuda:
                    torch.cuda.synchronize()
                times.append(time.time() - t0)
                model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None
            if cuda and peak > 0.92 * total_mem / 2 ** 30:
                print(f"[sft {cfg.name}] micro-batch {bs}: peak {peak:.1f} GB too close to the limit", flush=True)
                continue
            sec = sum(times[1:]) / max(len(times) - 1, 1) if len(times) > 1 else times[0]
            return {"per_device_bs": bs, "grad_accum": accum, "sec_per_step": sec, "peak_mem_gb": peak,
                    "probe_times": times}
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            print(f"[sft {cfg.name}] OOM at micro-batch {bs}; trying {bs // 2 or 'none'}", flush=True)
    raise RuntimeError("Out of memory even at micro-batch 1. Lower QWEN_MAX_PIXELS in settings.py.")


# ----------------------------------------------------------------------------- the run
def run_training(model, processor, cfg: SFTConfig, train_ex, eval_ex, work, deadline, device, dtype):
    """Train ``model`` (already LoRA-wrapped). Returns the summary dict."""
    import torch
    from transformers import get_cosine_schedule_with_warmup
    work = Path(work)
    log_dir = work / "logs"
    adapter_dir = work / "adapters" / cfg.name
    collator = SFTCollator(processor)
    ctx = autocast_for(dtype)
    t_start = time.time()

    sizing = probe(model, collator, train_ex, cfg, device, dtype, ctx)
    bs, accum = sizing["per_device_bs"], sizing["grad_accum"]
    steps_per_epoch = max(1, len(train_ex) // (bs * accum))
    cap = max(1, int(steps_per_epoch * cfg.epochs))
    if cfg.max_steps:
        cap = min(cap, cfg.max_steps)
    if deadline:
        affordable = int((deadline - time.time() - SIZING_MARGIN_S) / (sizing["sec_per_step"] * STEP_OVERHEAD))
        max_steps = max(1, min(cap, affordable))
    else:
        max_steps = cap
    eval_every = cfg.eval_every or max(50, max_steps // 8)
    sizing.update({"config": cfg.name, "examples": len(train_ex), "steps_per_epoch": steps_per_epoch,
                   "cap_steps": cap, "max_steps": max_steps, "eval_every": eval_every,
                   "fraction_of_epoch": round(max_steps / steps_per_epoch, 3),
                   "examples_seen": max_steps * bs * accum})
    dump_json(log_dir / f"sft_{cfg.name}_sizing.json", sizing)
    print(f"[sft {cfg.name}] sizing: {sizing['sec_per_step']:.1f} s/step, batch {bs}x{accum}, "
          f"{max_steps} steps ({sizing['fraction_of_epoch']:.0%} of an epoch)", flush=True)
    if max_steps < cfg.warn_below_steps:
        print(f"[sft {cfg.name}] WARNING: only {max_steps} steps fit the time budget "
              f"(< {cfg.warn_below_steps}); the adapter will be under-trained", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(params, cfg)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(cfg.warmup_ratio * max_steps)), max_steps)
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    history_path = log_dir / f"sft_{cfg.name}_history.csv"
    fields = ["step", "loss", "eval_loss", "lr", "grad_norm", "elapsed_min", "peak_mem_gb", "oom_skips"]
    hist = history_path.open("w", newline="")
    writer = csv.DictWriter(hist, fieldnames=fields)
    writer.writeheader()

    def log_row(**kw):
        writer.writerow({k: kw.get(k) for k in fields})
        hist.flush()

    best = {"eval_loss": math.inf, "step": 0}
    saved = False

    def maybe_save(step, eval_loss):
        nonlocal saved
        if eval_loss < best["eval_loss"] or not saved:
            best.update(eval_loss=eval_loss, step=step)
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            saved = True
            print(f"[sft {cfg.name}] saved adapter (step {step}, eval loss {eval_loss:.4f})", flush=True)

    gen = batches_forever(train_ex, collator, bs, S.SEED, cfg.num_workers)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_loss, run_n, nonfinite, oom_skips, stop_reason = 0.0, 0, 0, 0, "completed"
    t_loop = time.time()
    step = 0
    while step < max_steps:
        if deadline and time.time() > deadline - EARLY_STOP_MARGIN_S:
            stop_reason = "deadline"
            break
        try:
            step_loss = 0.0
            for _ in range(accum):
                loss_sum, n = forward_loss(model, next(gen), device, dtype, ctx)
                loss = loss_sum / max(n, 1) / accum
                if not torch.isfinite(loss):
                    nonfinite += 1
                    if nonfinite >= MAX_NONFINITE_MICROBATCHES:
                        raise FloatingPointError("Loss was non-finite for too many micro-batches in a row.")
                    continue
                nonfinite = 0
                (scaler.scale(loss) if scaler else loss).backward()
                step_loss += float(loss.detach())
            if scaler:
                scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError:
            oom_skips += 1
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"[sft {cfg.name}] OOM at step {step}: skipped ({oom_skips}/{MAX_OOM_SKIPS})", flush=True)
            if oom_skips >= MAX_OOM_SKIPS:
                stop_reason = "oom"
                break
            continue
        step += 1
        run_loss += step_loss
        run_n += 1
        if step % 10 == 0 or step == 1:
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
            log_row(step=step, loss=round(run_loss / run_n, 5), lr=scheduler.get_last_lr()[0],
                    grad_norm=round(grad_norm, 4), elapsed_min=round((time.time() - t_loop) / 60, 2),
                    peak_mem_gb=round(peak, 2) if peak else None, oom_skips=oom_skips)
            print(f"[sft {cfg.name}] step {step}/{max_steps} loss {run_loss / run_n:.4f} "
                  f"lr {scheduler.get_last_lr()[0]:.2e} gn {grad_norm:.2f} "
                  f"{(time.time() - t_loop) / step:.1f}s/step", flush=True)
            run_loss, run_n = 0.0, 0
        if step % eval_every == 0 and step < max_steps:
            ev = evaluate(model, collator, eval_ex, bs, device, dtype, ctx)
            log_row(step=step, eval_loss=round(ev, 5))
            print(f"[sft {cfg.name}] step {step} eval loss {ev:.4f}", flush=True)
            maybe_save(step, ev)
    final_eval = evaluate(model, collator, eval_ex, bs, device, dtype, ctx) if step > 0 else math.inf
    log_row(step=step, eval_loss=round(final_eval, 5) if math.isfinite(final_eval) else None)
    print(f"[sft {cfg.name}] final eval loss {final_eval:.4f} after {step} steps ({stop_reason})", flush=True)
    if step > 0 and math.isfinite(final_eval):
        maybe_save(step, final_eval)
    hist.close()
    summary = {"config": cfg.as_dict(), "stop_reason": stop_reason, "steps_done": step, "max_steps": max_steps,
               "best_eval_loss": best["eval_loss"] if math.isfinite(best["eval_loss"]) else None,
               "best_step": best["step"], "final_eval_loss": final_eval if math.isfinite(final_eval) else None,
               "hours": round((time.time() - t_start) / 3600, 3), "sec_per_step": sizing["sec_per_step"],
               "per_device_bs": bs, "grad_accum": accum, "fraction_of_epoch": round(step / steps_per_epoch, 3),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device.type == "cuda" else None,
               "adapter": str(adapter_dir) if saved else None}
    dump_json(log_dir / f"sft_{cfg.name}_summary.json", summary)
    if not saved:
        raise RuntimeError(f"No adapter was saved (stop reason: {stop_reason}).")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--media", type=Path, required=True)
    ap.add_argument("--deadline", type=float, default=0, help="unix time by which training must be finished")
    args = ap.parse_args(argv)
    cfg = CONFIGS[args.config]
    work = args.workdir
    (work / "logs").mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from ..backends.qwen import load_qwen

        torch.manual_seed(S.SEED)
        train_ex, skipped = build_examples(data.train_rows(cfg.train_rows_per_track), args.media,
                                           shuffle_seed=S.SEED)
        eval_rows = data.eval_loss_rows()
        if cfg.eval_rows_per_track:
            eval_rows = {t: r[:cfg.eval_rows_per_track] for t, r in eval_rows.items()}
        eval_ex, _ = build_examples(eval_rows, args.media)
        print(f"[sft {cfg.name}] {len(train_ex)} train / {len(eval_ex)} eval examples ({skipped} skipped)", flush=True)

        model, processor, dtype = load_qwen()
        model = prepare_lora(model, cfg, work / "logs")
        collator = SFTCollator(processor)
        for ex in train_ex[:2]:                       # show exactly what is trained on
            b = collator([ex])
            lab = b["labels"][0]
            shown = processor.tokenizer.decode(lab[lab != -100])
            print(f"[sft {cfg.name}] TRAIN TARGET: {shown!r}", flush=True)
            squash = lambda t: "".join(t.split())          # tokenizer decoding may normalise whitespace only
            assert squash(shown) == squash(ex["answer"] + "<|im_end|>"), f"label masking mismatch: {shown!r}"
        (work / "sft").mkdir(exist_ok=True)
        (work / "sft" / f"ready_{cfg.name}").write_text("ready")
        summary = run_training(model, processor, cfg, train_ex, eval_ex, work, args.deadline,
                               torch.device("cuda"), dtype)
        print(f"[sft {cfg.name}] done: {json.dumps(summary)[:400]}", flush=True)
        return 0
    except BaseException as exc:                      # a failed config must not take the other one down
        import traceback
        dump_json(work / "logs" / f"sft_{cfg.name}_failed.json",
                  {"config": cfg.name, "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc()})
        (work / "sft").mkdir(exist_ok=True)
        (work / "sft" / f"ready_{cfg.name}").write_text("failed")      # unblocks the staggered start
        print(f"[sft {cfg.name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
