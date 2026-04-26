**Design Description:** Straight-through container clipping only — replace `container = container.clamp(-1.0, 1.0)` with an STE that preserves the forward clip but passes gradients through saturated pixels. No EMA. No other change to loss, curriculum, INN, or optimizer.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Goal

Isolate the gradient-flow effect of straight-through container clipping from the weight-averaging effect of design001. Expectation: a smoother encoder optimization landscape near the [-1, 1] boundary, manifesting as either modestly higher PSNR (less over-embedding) or modestly higher bit_acc, with `bit_acc_clean = 1.000` preserved at every eval resolution. Forward values are byte-identical to baseline; only backward gradients change.

## 2. CLI flags / config for `baseline/exp0_inn_train.py`

This section enumerates the full training-run config (all CLI / config-level knobs).

Identical to baseline canonical sbatch invocation. **No locked flag changed. No new CLI flags.**

```
--num_blocks 8
--subnet_type light
--lr 4e-4
--img_weight 1.0
--bit_weight 20.0
--stage_a_steps 5000
--stage_b_steps 10000
--batch_size 16        # locked
--secret_len 896       # locked
--resolution 256       # locked
--max_steps 30000      # locked
--total_steps 30000    # locked
```

## 3. Files Builder may modify

- `code/exp0_inn_train.py`

Builder must **not** touch any of:
- `code/inn_model.py`, `code/losses.py` (no model or loss change)
- `infra/**` (locked)

Note: `_run_validation()` in this same file *also* calls `.clamp(-1, 1)` on the container (line 158), but that path is `@torch.no_grad()` so the STE has no semantic effect there. Builder must therefore **leave the validation-path clamp unchanged** (do not introduce STE in eval/validation paths — the locked `infra/exp0_inn_eval.py` also uses hard clip, and the train-time validation should mirror what eval will see at end-of-training).

## 4. Algorithmic spec — exact changes to `code/exp0_inn_train.py`

### 4.1 Add an STE helper near the top of the file (after `_psnr`, before `class NoiseCurriculum`)

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

### 4.2 Replace the training-loop hard clip

In the `while global_step < args.max_steps:` block, change line 371 from:

```python
container = container.clamp(-1.0, 1.0)
```

to:

```python
# STE clip — forward identical to clamp(-1,1), backward passes
# gradient through the saturated set so the encoder can learn from
# the bit-loss signal on over-embedded pixels (idea004/design002).
container = ste_clamp(container, -1.0, 1.0)
```

This is the *only* substantive line change in the training loop.

### 4.3 Do **not** modify

- The `_run_validation` function's `.clamp(-1, 1)` (line 158): `@torch.no_grad()` context, so STE is a no-op semantically; keep the hard clamp to mirror eval.
- The optimizer, scheduler, AMP, grad-clip, loss formulation, curriculum, model, dataset, attack module — all unchanged.
- The checkpoint payload — unchanged from baseline (no new keys).

## 5. Logging spec

Add one diagnostic field to the per-step `rec` dict (around line 413), measured under `torch.no_grad()` in the existing diagnostics block:

```python
sat_frac=float((container_f.abs() >= 1.0 - 1e-6).float().mean().item()),
```

This is the fraction of container pixels at or beyond the [-1, 1] boundary, i.e. the fraction of pixels for which baseline's hard clamp would have killed the gradient and STE now propagates it. Useful for:
- diagnosing whether STE is doing anything (if `sat_frac ≈ 0` throughout training, the gradient-flow change has no surface to act on and the design's null result is unsurprising);
- detecting runaway over-embedding (if `sat_frac` grows unbounded under STE, the encoder is exploiting the now-flowing gradient to push the unconstrained pre-clip residual to large magnitudes — the L1 host loss should counteract this, but it is the failure mode to watch).

No other logging change. `metrics.jsonl` schema = baseline keys ∪ `{sat_frac}`.

## 6. Expected behaviour after the change

- Training cost: zero. STE adds one elementwise add and one elementwise sub at each step, both fused with the existing clamp; well below the noise floor of step time. Total walltime within ±5% of baseline.
- Forward outputs (container values) are byte-identical to baseline at fp32; under AMP they are within fp16 rounding of baseline.
- `metrics.jsonl` per-step keys = baseline keys ∪ `{sat_frac}`. Expected `sat_frac` magnitude: 1e-4 to 1e-2 in the clean phase (rising as the encoder learns to embed) and 1e-3 to 5e-2 in the full-attack phase. If `sat_frac > 0.1` sustained, the design is in the runaway regime — Builder should not "fix" this but should report it; it is a finding, not a bug.
- `ckpt_latest.pt` payload schema = identical to baseline.
- Eval `metrics.csv`: target `score >= 27.5` (i.e. >= baseline + ~0.25). Hard condition: `bit_acc_clean = 1.000` at every one of the 9 resolutions.

## 7. Invariants Builder must preserve

1. **Locked flags unchanged:** `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.
2. **Locked infra unchanged:** zero edits under `infra/**`. STE is implemented in the training script; the eval script (locked) continues to consume `ckpt["model"]` and to apply its own hard clamp on the container at eval time.
3. **Hard condition:** `bit_acc_clean = 1.000` at every eval resolution. STE never alters forward values, so the forward behaviour at eval time (which uses hard clamp anyway in `infra/exp0_inn_eval.py`) is unchanged from a baseline trained for the same number of steps with the same seed; the only difference is the trajectory through weight space.
4. **No mode confusion:** STE applies only on the training-loop forward path (one call site, line 371). The validation path's hard clamp inside `_run_validation` and the eval-time hard clamp inside the locked `infra/exp0_inn_eval.py` are intentionally left unchanged.
5. **No new CLI flags, no new module-level state, no checkpoint format change.**

## 8. Edge cases

- **Pre-clamp magnitude:** if the INN's pre-clip output is e.g. 1.5 (a 50% overshoot), the baseline gradient at that pixel is 0; under STE it equals the upstream gradient. The L1 host loss will then see a `(container - cover)` term that pulls the *unconstrained* pre-clip residual back toward `cover` (which is in [-1, 1]), effectively coupling the L1 loss to over-embedding magnitude in a way the baseline cannot — this is the intended mechanism. The bit loss likewise sees a useful signal on those pixels.
- **Numerical stability:** at fp16 inside autocast, `x.clamp(lo, hi) - x` may underflow to 0 for small overshoots, in which case STE degenerates to identity (correct fallback). For overshoots large enough to matter, the subtraction is well-conditioned.
- **`container.abs() >= 1 - 1e-6`** for `sat_frac` is intentionally on the post-STE tensor (which has the clamp already applied); this measures the same "saturated set" as the baseline clamp would have produced.
- **Walltime sanity-check:** Builder should compare `wallclock_s` at step 30000 against baseline (~4–5 h on one GPU) and flag if it exceeds 1.1× — that would indicate an unintended graph rebuild every step from `.detach()` not being recognised as constant by the autograd engine, which would be a bug to investigate.
