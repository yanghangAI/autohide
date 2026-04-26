**Design Description:** Multi-scale Sobel JND mask + complementary edge-aware TV penalty (combined). Stack design001 (multi-scale Sobel mask, σ ∈ {1.0, 2.0, 4.0}, β=4.0, 99th-percentile per-image normalize, max-aggregate across scales) and design002 (TV term `tv_w * mean((1 - w_norm) · |∇(stego - cover)|_luma)` with `tv_w = 0.3`). The multi-scale `w_multi` is reused everywhere — YUV term, masked-L1 term, and the TV term's `w_norm`. All other parent settings preserved.

**Parent:** runs/idea003/design003

**Starting Point:** runs/idea003/design003

---

## 1. Goal and motivation

This is direction (3) of `runs/idea007/idea.md` — the combined design that stacks design001 and design002. Hypothesis: if the two interventions act on independent failure modes (mask precision vs explicit flat-region prohibition), they should add. If they overlap, this design quantifies the interaction so iter-4 can pick the cheaper of the two (design001 is mask-only, design002 is TV-only; design003 pays for both).

Both subcomponents are specified in their respective design files (`runs/idea007/design001/design.md` for the multi-scale mask, `runs/idea007/design002/design.md` for the TV term). This design takes both verbatim, with the additional requirement that the TV term's `w_norm` is computed from the *multi-scale* `w` (not from a single-scale mask). This is the natural composition — both components share the same mask object.

Parent is `runs/idea003/design003` (not one of the iter-7 designs) because the per-iteration evaluation matrix wants three independent comparisons against the same baseline; design003 measures the additive effect of the two new components against the current best `Done`.

If `score > max(score_d001, score_d002)`, the components are additive and design003 is the new best. If `score ≈ score_d001 ≈ score_d002`, they are redundant. If `score < parent`, the combined intervention over-constrains the encoder.

## 2. Files changed (Builder may only edit these)

Builder may only touch the following files inside `code/` (mirroring `baseline/`):

- `code/losses.py` — apply BOTH the design001 edits (add `_gaussian_kernel_1d`, `_gaussian_blur_luma`, `multiscale_sobel_jnd_mask`, extend `jnd_mode` set with `"sobel_multiscale"`, route mask compute through it) AND the design002 edits (add `_sobel_mag_luma`, `tv_w` and `tv_eps` constructor kwargs, the TV term inside `forward`, `l_tv` in `logs`).
- `code/exp0_inn_train.py` — pass `jnd_mode="sobel_multiscale"`, `jnd_sigmas=(1.0, 2.0, 4.0)`, `jnd_pct=0.99`, AND `tv_w=0.3` to `ImageSecretLoss(...)`; emit `jnd_mode="sobel_multiscale"`, `jnd_sigmas=[1.0, 2.0, 4.0]`, `jnd_pct=0.99`, `l_tv`, and `tv_w=0.3` in the per-step `metrics.jsonl` record.

Untouched (locked or out of scope):
- All of `infra/**` (byte-locked).
- `code/inn_model.py` — no architecture change.
- `code/train.py` — unchanged.

## 3. CLI flag values for `baseline/exp0_inn_train.py`

Locked (must NOT change): `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.

Tuned (identical to parent `runs/idea003/design003`):
- `--num_blocks 16`
- `--subnet_type rdb`
- `--lr 2e-4`
- `--img_weight 1.0`
- `--bit_weight 20.0`
- `--stage_a_steps 5000`
- `--stage_b_steps 10000`

No new CLI flags. Mask config (`jnd_beta=4.0`, `pixel_w=0.5`, `jnd_mode="sobel_multiscale"`, `jnd_sigmas=(1.0, 2.0, 4.0)`, `jnd_pct=0.99`) and TV config (`tv_w=0.3`) are all hard-coded in the `ImageSecretLoss(...)` constructor call inside `code/exp0_inn_train.py`.

## 4. Algorithmic spec (concrete)

### 4.1 `code/losses.py`

Apply BOTH sets of edits.

**From design001 (verbatim, see `runs/idea007/design001/design.md` §4.1):**
- Add `_gaussian_kernel_1d(sigma, device, dtype)`.
- Add `_gaussian_blur_luma(luma, sigma)` (separable 2D Gaussian, replicate padding).
- Add `multiscale_sobel_jnd_mask(cover, sigmas=(1.0, 2.0, 4.0), pct=0.99)` decorated with `@torch.no_grad`. Per-scale 99th-percentile normalize, then per-pixel max-aggregate across scales.
- Extend `jnd_mode` assert in `ImageSecretLoss.__init__` to accept `"sobel_multiscale"`.
- Add `jnd_sigmas=(1.0, 2.0, 4.0)` and `jnd_pct=0.99` constructor kwargs (stored as `self.jnd_sigmas`, `self.jnd_pct`).
- In `forward`, dispatch the mask compute on `jnd_mode`: `"sobel"` → `sobel_jnd_mask`, `"variance"` → `variance_jnd_mask`, `"sobel_multiscale"` → `multiscale_sobel_jnd_mask(cover, sigmas=self.jnd_sigmas, pct=self.jnd_pct)`.

**From design002 (verbatim, see `runs/idea007/design002/design.md` §4.1):**
- Add `_sobel_mag_luma(rgb)` returning differentiable Sobel magnitude on the luma channel of an RGB tensor.
- Add `tv_w=0.0` and `tv_eps=1e-6` constructor kwargs (stored as `self.tv_w`, `self.tv_eps`).
- Inside the `if self.jnd_beta > 0.0:` branch in `forward`, after `w` is computed and `l_pixel` is computed, add the TV term using the *same* `w`:

  ```python
  if self.tv_w > 0.0:
      B = w.shape[0]
      w_max = w.view(B, -1).amax(dim=1).clamp_min(self.tv_eps)
      w_norm = w / w_max.view(B, 1, 1, 1)        # (B, 1, H, W) in [0, 1]
      comp = 1.0 - w_norm
      g_res = _sobel_mag_luma(stego - cover)      # differentiable
      l_tv = (comp * g_res).mean()
  else:
      l_tv = torch.zeros((), device=stego.device, dtype=stego.dtype)
  ```

  In the `else:` branch (jnd_beta = 0), use the `l_tv = g_res.mean()` fallback as in design002.
- Add `+ self.tv_w * l_tv` to `l_quality`.
- Add `l_tv=l_tv.detach()` to the `logs` dict.

**Critical composition detail:** `w` here is the multi-scale `w` (since `jnd_mode="sobel_multiscale"`). The TV term's `w_norm` is therefore derived from the multi-scale mask, not a single-scale one. This is the intended composition — a single mask object is shared by all three terms (YUV, L1 pixel, TV).

The combined `__init__` signature:

```python
def __init__(self,
             secret_weight: float = 20.0,
             yuv_w: float = 1.5,
             lpips_w: float = 1.0,
             ffl_w: float = 1.5,
             jnd_beta: float = 0.0,
             pixel_w: float = 0.0,
             jnd_mode: str = "sobel",
             jnd_sigmas: tuple[float, ...] = (1.0, 2.0, 4.0),
             jnd_pct: float = 0.99,
             tv_w: float = 0.0,
             tv_eps: float = 1e-6):
    super().__init__()
    assert jnd_mode in ("sobel", "variance", "sobel_multiscale"), f"unknown jnd_mode={jnd_mode}"
    self.secret_weight = secret_weight
    self.yuv_w = yuv_w
    self.lpips_w = lpips_w
    self.ffl_w = ffl_w
    self.jnd_beta = jnd_beta
    self.pixel_w = pixel_w
    self.jnd_mode = jnd_mode
    self.jnd_sigmas = tuple(jnd_sigmas)
    self.jnd_pct = float(jnd_pct)
    self.tv_w = tv_w
    self.tv_eps = tv_eps
    self.lpips = lpips_lib.LPIPS(net="alex", verbose=False)
    for p in self.lpips.parameters():
        p.requires_grad_(False)
    self.ffl = FocalFrequencyLoss()
```

### 4.2 `code/exp0_inn_train.py`

Modify the `ImageSecretLoss(...)` call:

```python
img_loss_mod = ImageSecretLoss(
    secret_weight=args.bit_weight,   # 20.0
    yuv_w=1.5,
    lpips_w=0.0,
    ffl_w=1.5,
    jnd_beta=4.0,
    pixel_w=0.5,
    jnd_mode="sobel_multiscale",
    jnd_sigmas=(1.0, 2.0, 4.0),
    jnd_pct=0.99,
    tv_w=0.3,
).to(device)
```

In the per-step `metrics.jsonl` record, set `jnd_mode="sobel_multiscale"` and add the new descriptor / metric fields:

```python
jnd_mode="sobel_multiscale",
jnd_sigmas=[1.0, 2.0, 4.0],
jnd_pct=0.99,
l_tv=float(loss_logs["l_tv"]),
tv_w=0.3,
```

(Keep all existing fields: `l_yuv`, `l_yuv_unweighted`, `l_pixel`, `jnd_beta=4.0`, `pixel_w=0.5`.)

No other change to `exp0_inn_train.py`.

## 5. Expected behavior after the change

- `metrics.jsonl` per-step records include `jnd_mode="sobel_multiscale"`, `jnd_sigmas=[1.0, 2.0, 4.0]`, `jnd_pct=0.99`, `l_tv`, `tv_w=0.3`, plus the parent's existing `l_yuv`, `l_yuv_unweighted`, `l_pixel`, `jnd_beta=4.0`, `pixel_w=0.5`.
- `l_yuv`, `l_pixel`, `l_tv` magnitudes: as in designs 001 and 002 (combined). The contribution of each to `l_quality` is unchanged from those designs because the loss formula is purely additive in the new term.
- Hypothesis on outcome: best case `score > max(score_d001, score_d002)` (additive); typical case `score ≈ best of d001/d002` (overlapping); worst case `score < parent` (over-constraint). Even the worst case stays bounded by the parent's safety profile per the idea spec — both subcomponents are softer than (e.g.) adding a deep perceptual term.
- Wallclock: design001's three Gaussian blurs + three Sobel passes + three quantile-99 reductions, plus design002's one residual Sobel pass. Total <2.5 ms/step on a 2080Ti; well within parent's walltime headroom.
- A design is `Done` when training reaches `step >= 30000`. After training, the locked multi-resolution eval (`infra/exp0_inn_eval.py`, 13 attacks × 9 resolutions) runs and writes `eval.json` → aggregated into `metrics.csv`.

## 6. Constraints and invariants Builder must preserve

- All constraints from design001 §6 apply (multi-scale mask correctness, per-scale-then-aggregate normalization order, mask detached, mask from `cover` only).
- All constraints from design002 §6 apply (TV term on `(stego - cover)` not on `stego`, residual Sobel must be differentiable, `w_norm` per-image normalization).
- One mask compute per `forward` — the multi-scale mask is computed once and the same `w` is reused for the YUV term, the L1 pixel term, and the TV term's `w_norm`. Computing the mask twice doubles helper cost.
- All loss-construction kwargs identical to the parent except for the *added* `jnd_mode="sobel_multiscale"` (replaces `"sobel"`), `jnd_sigmas`, `jnd_pct`, and `tv_w=0.3`. Specifically: `secret_weight=20.0`, `yuv_w=1.5`, `lpips_w=0.0`, `ffl_w=1.5`, `jnd_beta=4.0`, `pixel_w=0.5` are unchanged.
- No edits to `infra/**`. No edits to `code/inn_model.py` or `code/train.py`.
- Locked CLI flags untouched.
- All non-locked CLI flags identical to the parent.
- LPIPS off (`lpips_w=0.0`); short-circuit preserved.
- Existing `sobel_jnd_mask` and `variance_jnd_mask` helpers MUST NOT be deleted or modified.
- Optimizer, LR scheduler, AMP autocast scope, gradient clipping, curriculum α-ramp, validation cadence, checkpoint format unchanged.
- `bit_acc_clean = 1.000` at every eval resolution remains the hard target. Combined intervention has the largest deviation surface from the parent — if a smoke run shows `bit_acc_clean_min < 0.99` at step 5000+, flag to Designer (would suggest reducing `tv_w` first since the TV term is the harder structural prohibition).

## 7. Edge cases

- **Composition correctness**: the multi-scale `w` has the same shape `(B, 1, H, W)` and value range `(0, 1]` as the single-scale `w`. The TV term's `w_norm = w / w.amax` works identically on both. No special-casing needed.
- **`w_norm` derived from multi-scale mask**: with multi-scale max-aggregation, `M_multi.amax` per image is typically ≈ 1 (since 99th-percentile normalization sets that scale). `w_min = 1/(1+β·M_multi.amax) ≈ 0.2` at β=4; `w_max = 1` at the flattest pixel. `w_norm ∈ [0.2, 1]`, `comp ∈ [0, 0.8]`. Same regime as design002.
- **Constant-luma cover**: all multi-scale `M` maps are 0 → `M_multi = 0` → `w = 1` → `w_norm = 1` → `comp = 0` → `l_tv = 0`. Correct degenerate behavior.
- **AMP / autocast**: combination of design001's fp32-cast for `torch.quantile` and design002's `+1e-12` in the residual Sobel is sufficient. No additional precision concerns.
- **`metrics.jsonl` key stability**: `jnd_sigmas` logged as a Python list, `l_tv` as a float. Same JSON-serializability rules as the individual designs.
- **Test for pure additivity vs interaction**: comparing `score_d003 - score_parent` against `(score_d001 - score_parent) + (score_d002 - score_parent)` quantifies the interaction. This is a Designer-side analysis after results land, not a Builder concern, but the design is intentionally constructed so the comparison is well-defined — both subcomponents share the same parent and same mask object.
