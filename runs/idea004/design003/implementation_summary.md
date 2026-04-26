**Files changed:**
code/exp0_inn_train.py

**Changes:**
- `code/exp0_inn_train.py`: Combined the design001 (EMA shadow weights, decay 0.999, warmup at `stage_a_steps`, dual-slot checkpoint format) and design002 (straight-through container clipping) interventions into a single training script. The `WeightEMA` class and `ste_clamp` helper are added near the top of the file (after `_psnr`, before `NoiseCurriculum`), the EMA is constructed and updated per design001, the training-loop forward path uses `ste_clamp` instead of `.clamp(-1, 1)` per design002, `save_ckpt()` writes EMA weights to the eval-consumed `model` slot with online weights preserved in `model_online`, the resume block restores online weights and EMA shadow, and the per-step `metrics.jsonl` records gain both `ema_active` and `sat_frac` diagnostic fields. The validation-path hard clamp at line ~158 is intentionally left unchanged.

In `code/exp0_inn_train.py`, the EMA helper class:

```python
class WeightEMA:
    """Fp32 exponential-moving-average shadow of a model's parameters.

    - Holds one fp32 buffer per parameter on the same device as the param.
    - update() runs *outside* autocast in fp32 to avoid drift from low-
      precision averaging.
    - Buffers (BatchNorm running stats, etc.) are copied verbatim every
      step (no averaging) — INNCodec has no BN but this future-proofs.
    - state_dict() / load_state_dict() round-trip the shadow tensors plus
      the integer `num_updates` counter so resume is bit-exact.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters() if p.requires_grad
        }
        self.buffers = {
            name: b.detach().clone()
            for name, b in model.named_buffers()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)
        for name, b in model.named_buffers():
            self.buffers[name].copy_(b.detach())
        self.num_updates += 1
```

In `code/exp0_inn_train.py`, the STE helper:

```python
def ste_clamp(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    """Straight-through estimator for hard clipping.

    Forward: identical to `x.clamp(lo, hi)` (byte-identical for finite x).
    Backward: gradient passes through unchanged on the entire input
    domain — including the saturated set where the standard clamp
    derivative is zero.

    Implemented as `x + (x.clamp(lo, hi) - x).detach()`:
      - `(x.clamp - x).detach()` is a constant offset wrt autograd.
      - The autograd graph sees only the leading `x`, so d(out)/d(x) = 1
        everywhere.
      - Forward value = `x + clamp(x) - x` = `clamp(x)`.

    Robust to fp16 / autocast: the cast happens before clamp, the .detach()
    severs the graph at the right point, and the operation is allocation-
    free in steady state.
    """
    return x + (x.clamp(lo, hi) - x).detach()
```

In `code/exp0_inn_train.py`, EMA constants in `main()`:

```python
    EMA_DECAY = 0.999
    EMA_WARMUP_STEPS = args.stage_a_steps  # start EMA only after the clean phase
```

In `code/exp0_inn_train.py`, EMA construction:

```python
    ema = WeightEMA(model, decay=EMA_DECAY)
    print(f"[ema] enabled: decay={EMA_DECAY} warmup_steps={EMA_WARMUP_STEPS}")
```

In `code/exp0_inn_train.py`, the training-loop forward call site (STE clip):

```python
            container = model.embed(cover, bits)
            # STE clip + EMA shadow weights (idea004/design003 — combined).
            container = ste_clamp(container, -1.0, 1.0)
            attacked = attack_module(container) if attack_module is not None else container
```

In `code/exp0_inn_train.py`, the EMA update inside the training loop:

```python
        if global_step >= EMA_WARMUP_STEPS:
            ema.update(model)
```

In `code/exp0_inn_train.py`, the dual-slot `save_ckpt()` payload:

```python
        payload = {
            "model": eval_state,                 # consumed by eval (EMA after warmup)
            "model_online": model.state_dict(),  # for resume
            "ema": ema.state_dict(),
```

In `code/exp0_inn_train.py`, the resume block:

```python
        if "model_online" in ck:
            model.load_state_dict(ck["model_online"])
        else:
            model.load_state_dict(ck["model"])
        if "ema" in ck:
            ema.load_state_dict(ck["ema"])
            print(f"[resume] EMA restored: num_updates={ema.num_updates}")
        else:
            ema = WeightEMA(model, decay=EMA_DECAY)
            print("[resume] no EMA in ckpt; reinitialised from current online weights")
```

In `code/exp0_inn_train.py`, the `sat_frac` diagnostic computation:

```python
                sat_frac = float((container_f.abs() >= 1.0 - 1e-6).float().mean().item())
```

In `code/exp0_inn_train.py`, the new logging fields added to the per-step record:

```python
                       ema_active=int(global_step >= EMA_WARMUP_STEPS),
                       sat_frac=sat_frac,
```
