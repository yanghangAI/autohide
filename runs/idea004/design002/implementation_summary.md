**Files changed:**
code/exp0_inn_train.py

**Changes:**
- `code/exp0_inn_train.py`: Added an `ste_clamp` helper near the top of the file (after `_psnr`, before `NoiseCurriculum`) that implements `x + (x.clamp(lo, hi) - x).detach()` so the forward value is byte-identical to a hard clamp but the backward pass propagates the upstream gradient through the saturated set. Replaced the single training-loop call `container = container.clamp(-1.0, 1.0)` (line ~371) with `container = ste_clamp(container, -1.0, 1.0)`. The `_run_validation` hard clamp at line 158 is intentionally left unchanged (it runs under `@torch.no_grad()` and must mirror the locked eval-time hard clamp). Also added a per-step `sat_frac` diagnostic to `metrics.jsonl` to measure the fraction of saturated container pixels.

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

In `code/exp0_inn_train.py`, the training-loop call site:

```python
            container = model.embed(cover, bits)
            # STE clip — forward identical to clamp(-1,1), backward passes
            # gradient through the saturated set so the encoder can learn from
            # the bit-loss signal on over-embedded pixels (idea004/design002).
            container = ste_clamp(container, -1.0, 1.0)
            attacked = attack_module(container) if attack_module is not None else container
```

In `code/exp0_inn_train.py`, the `sat_frac` diagnostic and its addition to the per-step log record:

```python
                sat_frac = float((container_f.abs() >= 1.0 - 1e-6).float().mean().item())
```

In `code/exp0_inn_train.py`, the `sat_frac` field added to the per-step record:

```python
                       sat_frac=sat_frac,
```
