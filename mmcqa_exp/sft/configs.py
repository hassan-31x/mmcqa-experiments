"""SFT configurations. S1 and S2 differ only in the learning rate."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class SFTConfig:
    name: str
    lr: float
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    per_device_bs: int = 4
    grad_accum: int = 4               # effective batch = per_device_bs * grad_accum = 16
    epochs: float = 1.0
    max_steps: int = None             # hard cap on optimizer steps (smoke runs)
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    eval_every: int = None            # default: max(50, max_steps // 8)
    num_workers: int = 2
    train_rows_per_track: int = None  # None = all 10,000
    eval_rows_per_track: int = None   # None = the 50-row slice
    min_lora_dim: int = 64            # skip Linear layers narrower than this (tiny gating projections)
    warn_below_steps: int = 300
    model: str = "4b"                 # "4b" = Qwen3.5-4B, "9b" = Qwen3.5-9B (both NF4)

    @property
    def effective_batch(self) -> int:
        return self.per_device_bs * self.grad_accum

    def as_dict(self) -> dict:
        return asdict(self)


S1 = SFTConfig("S1", lr=1e-4)
S2 = SFTConfig("S2", lr=2e-4)
SMOKE = SFTConfig("SMOKE", lr=1e-4, max_steps=10, eval_every=5, train_rows_per_track=64, eval_rows_per_track=8,
                  warn_below_steps=0)

SMOKE2 = SFTConfig("SMOKE2", lr=2e-4, max_steps=10, eval_every=5, train_rows_per_track=64, eval_rows_per_track=8,
                   warn_below_steps=0)

# Experiment A: the same recipe on the quantized 9B. max_steps=870 is what S2 actually trained (13,920 examples);
# if the time budget allows fewer steps the sizing step shrinks it and records the fraction of the 4B's data seen.
S1_9B = SFTConfig("S1_9B", lr=1e-4, model="9b", max_steps=870, warn_below_steps=0)
S2_9B = SFTConfig("S2_9B", lr=2e-4, model="9b", max_steps=870, warn_below_steps=0)

CONFIGS = {c.name: c for c in (S1, S2, SMOKE, SMOKE2, S1_9B, S2_9B)}
