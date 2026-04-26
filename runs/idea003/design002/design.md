**Design Description:** Variance-pooled JND mask on the YUV term — same `w = 1/(1 + β·M)` weighting as design001, but `M(cover)` is the per-image-normalized **local std-dev of luma over a 7×7 window** (computed via a uniform-kernel conv + the `E[X²]−E[X]²` square-then-conv trick) instead of Sobel-gradient magnitude. β=4.0. FFL term unchanged. All other design003 settings preserved.

**Parent:** runs/idea001/design003

**Starting Point:** runs/idea001/design003

---

## 1. Motivation and approach

This is direction (2) of `runs/idea003/idea.md`. design001 (in this idea) tests a **gradient/edge** mask; this design tests a **texture/variance** mask. Edges and texture are correlated but not identical:
- Soft gradient sky: low edge magnitude, low variance → both masks would weight ~1.
- Uniform-noise grass: moderate edge magnitude, high variance → the variance mask down-weights more strongly than the Sobel mask.
- Hard single edge between two flat regions: high edge magnitude, moderate variance (only in the 7×7 windows that straddle the edge) → the Sobel mask down-weights more strongly.

Variance-pooling captures **regional texture energy** rather than **point-edge intensity**, which is closer to the JND psychophysics literature (e.g. Watson's DCT-domain JND, Liu et al. 2010) where masking comes from local activity over a perceptual neighborhood, not from individual edges. Comparing design001 vs design002 isolates which geometric prior — gradient vs variance — better predicts perceptual capacity in this dataset.

All other design003 hyperparameters preserved: `yuv_w=1.5`, `ffl_w=1.5`, `lpips_w=0.0`, `secret_weight=20.0`, `--num_blocks 16`, `--subnet_type rdb`, `--lr 2e-4`, `--stage_a_steps 5000`, `--stage_b_steps 10000`. β=4.0 to match design001 (so the only difference between the two designs is the mask geometry, not its strength).

## 2. Files changed (Builder may only edit these)

- `code/baseline/losses.py` — add a `variance_jnd_mask(cover, win=7)` helper and a `jnd_mode` argument to `ImageSecretLoss.__init__` ("sobel" or "variance"); apply per-pixel weighting on the YUV term.
- `code/baseline/exp0_inn_train.py` — pass `jnd_beta=4.0, jnd_mode="variance"` to `ImageSecretLoss(...)`; add `l_yuv_unweighted`, `jnd_beta`, `jnd_mode` to the per-step `metrics.jsonl` record.

Untouched (locked): everything in `infra/**`, `code/baseline/inn_model.py`, `code/baseline/train.py`.

## 3. CLI flag values for `baseline/exp0_inn_train.py` (config)

Locked (must NOT change): `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.

Tuned (identical to design003 parent):
- `--num_blocks 16`
- `--subnet_type rdb`
- `--lr 2e-4`
- `--img_weight 1.0`
- `--bit_weight 20.0`
- `--stage_a_steps 5000`
- `--stage_b_steps 10000`

No new CLI flags; `jnd_beta=4.0, jnd_mode="variance"` are hard-coded in the `ImageSecretLoss(...)` instantiation in `exp0_inn_train.py`.

## 4. Algorithmic changes (concrete)

### 4.1 `code/baseline/losses.py` — add variance mask + dispatch

**Add module-level helpers** (in addition to or alongside the Sobel helper from design001 if both designs end up sharing this file in a future rebase; for this design as a standalone child of design003, only the variance helper is required):

```python
@torch.no_grad()
def variance_jnd_mask(cover: torch.Tensor, win: int = 7) -> torch.Tensor:
    """Per-image-normalized local std-dev of the luma channel over a `win`x`win`
    uniform window.  Input `cover` in [-1, 1], shape (B, 3, H, W).  Output
    shape (B, 1, H, W), values in [0, 1] (per-image max-normalized).

    Uses E[X^2] - E[X]^2 with two single-channel uniform-kernel convs.
    Returns a tensor detached from the autograd graph.
    """
    yuv = rgb_to_yuv(cover)
    luma = (yuv[:, 0:1] + 1.0) * 0.5  # (B,1,H,W) in [0,1]
    pad = win // 2
    k = torch.ones((1, 1, win, win), device=luma.device, dtype=luma.dtype) / float(win * win)
    mean = F.conv2d(luma, k, padding=pad)
    mean_sq = F.conv2d(luma * luma, k, padding=pad)
    var = (mean_sq - mean * mean).clamp_min(0.0)  # numerical guard
    std = torch.sqrt(var + 1e-12)                 # (B,1,H,W)
    B = std.shape[0]
    flat_max = std.view(B, -1).amax(dim=1).clamp_min(1e-6)
    std = std / flat_max.view(B, 1, 1, 1)
    return std.detach()
```

**Modify `ImageSecretLoss.__init__`** to accept `jnd_beta` and `jnd_mode`:

```python
def __init__(self,
             secret_weight: float = 20.0,
             yuv_w: float = 1.5,
             lpips_w: float = 1.0,
             ffl_w: float = 1.5,
             jnd_beta: float = 0.0,
             jnd_mode: str = "variance"):
    super().__init__()
    assert jnd_mode in ("sobel", "variance"), f"unknown jnd_mode={jnd_mode}"
    self.secret_weight = secret_weight
    self.yuv_w = yuv_w
    self.lpips_w = lpips_w
    self.ffl_w = ffl_w
    self.jnd_beta = jnd_beta
    self.jnd_mode = jnd_mode
    self.lpips = lpips_lib.LPIPS(net="alex", verbose=False)
    for p in self.lpips.parameters():
        p.requires_grad_(False)
    self.ffl = FocalFrequencyLoss()
```

**Modify `ImageSecretLoss.forward`**, replacing
```python
l_yuv = F.mse_loss(rgb_to_yuv(stego), rgb_to_yuv(cover))
```
with:
```python
yuv_stego = rgb_to_yuv(stego)
yuv_cover = rgb_to_yuv(cover)
sq = (yuv_stego - yuv_cover) ** 2  # (B, 3, H, W)
l_yuv_unweighted = sq.mean()
if self.jnd_beta > 0.0:
    if self.jnd_mode == "variance":
        M = variance_jnd_mask(cover, win=7)   # (B, 1, H, W) in [0, 1], detached
    else:  # "sobel"
        M = sobel_jnd_mask(cover)             # (B, 1, H, W) in [0, 1], detached
    w = 1.0 / (1.0 + self.jnd_beta * M)       # (B, 1, H, W), broadcast over chans
    l_yuv = (w * sq).mean()
else:
    l_yuv = l_yuv_unweighted
```

(Builder may include the `sobel_jnd_mask` helper from design001's spec for symmetry, but this design only exercises the `"variance"` branch. If only one helper is added, only the `variance` branch needs to compile/run.)

Add `l_yuv_unweighted` to the returned `logs` dict (same shape as design001 §4.1).

### 4.2 `code/baseline/exp0_inn_train.py` — pass mode + β

Modify the `ImageSecretLoss(...)` call:
```python
img_loss_mod = ImageSecretLoss(
    secret_weight=args.bit_weight,   # 20.0
    yuv_w=1.5,
    lpips_w=0.0,
    ffl_w=1.5,
    jnd_beta=4.0,
    jnd_mode="variance",
).to(device)
```

In the per-step `metrics.jsonl` record (the `rec = dict(...)` block), add:
```python
l_yuv_unweighted=float(loss_logs["l_yuv_unweighted"]),
jnd_beta=4.0,
jnd_mode="variance",
```

No other change to `exp0_inn_train.py`.

## 5. Expected behavior after the change

- `metrics.jsonl` per-step records gain `l_yuv_unweighted` (float), `jnd_beta` (4.0), and `jnd_mode` ("variance").
- `l_yuv` (weighted) at convergence should again be ~30–60% smaller than `l_yuv_unweighted`. Compared to design001 (Sobel), the variance mask tends to produce *smoother*, *broader* down-weighting regions (since std-dev over a 7×7 window blurs over fine edges), which means more pixels get partial down-weighting and fewer pixels get extreme down-weighting.
- `bit_acc_clean = 1.000` at every eval resolution.
- Hypothesis on outcome: PSNR (and `score`) at step 30000 ≥ design003 (27.77). If design002 > design001, regional texture energy is a better mask than point-edge intensity for this task; if design001 > design002, the encoder benefits from sharp localized down-weighting around edges. Either ordering is informative.
- Wallclock per step: design003 + ~1 ms (two 7×7 uniform convs on 16×256² fp32 ≈ <1 ms on 2080Ti).

## 6. Constraints and invariants Builder must preserve

- No edits to `infra/**`. No edits to `code/baseline/inn_model.py` or `code/baseline/train.py`.
- Locked CLI flags untouched.
- All non-locked CLI flags identical to design003 parent.
- LPIPS remains off (`lpips_w=0.0`); short-circuit preserved.
- `secret_weight=20.0` preserved.
- JND mask computed under `torch.no_grad()` from `cover` only and detached.
- Per-image normalization MUST be applied so β has consistent semantics across covers.
- The `var.clamp_min(0.0)` guard MUST be present — `mean_sq - mean*mean` can produce small negative values from floating-point cancellation on near-uniform regions; without the clamp `sqrt` produces NaN.
- Optimizer, scheduler, AMP autocast scope, gradient clipping, curriculum α-ramp, validation cadence, checkpoint format unchanged.
- `bit_acc_clean = 1.000` must remain achievable at convergence.

## 7. Edge cases

- **Constant-luma cover**: `var = 0` everywhere → `std = ~1e-6` everywhere → `flat_max` ≈ 1e-6 → after divide, `M` ≈ 1 everywhere → `w = 1/(1+4) = 0.2` everywhere → uniformly down-weighted YUV term. This is acceptable but slightly weaker than design001's behavior on a constant cover (where Sobel yields `M=0` and `w=1`); the difference is harmless because constant-luma covers are vanishingly rare in the COCO-derived training set, and the per-image weighting still preserves `bit_acc` (only the relative weighting matters for gradients).
- **Single-pixel impulse on otherwise constant cover**: only the 7×7 windows touching the impulse have non-zero variance; per-image max-normalize sets those windows' `M` to 1.0 and the rest to 0. Correct localized down-weighting.
- **AMP / autocast**: helper called inside the existing fp32-wrapped autocast=False block; all conv2d + sqrt run in fp32.
- **Channel broadcast**: same as design001 — `w` shape `(B,1,H,W)` broadcasts over the 3 YUV channels.
- **Padding**: `padding=3` for the 7×7 uniform kernel uses zero padding, which slightly under-estimates the local mean (and therefore over-estimates the local variance) on the 3-pixel border. Negligible relative to the 256² interior; per-image max-normalize absorbs the small bias.
- **Numerical**: `(mean_sq - mean*mean).clamp_min(0.0)` followed by `sqrt(... + 1e-12)` is a standard variance-from-uniform-conv recipe; the `1e-12` floor prevents `sqrt(0)` gradient issues even though the tensor is detached (defensive).
- **Window size choice**: `win=7` is fixed (idea text specifies 7×7). Builder MUST NOT vary this for this design; a `win` sweep is out of scope.
