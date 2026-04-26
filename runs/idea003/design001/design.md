**Design Description:** Sobel-gradient JND mask on the YUV term — replace `F.mse_loss(yuv(stego), yuv(cover))` with a per-pixel weighted MSE `mean(w * (yuv(stego)-yuv(cover))^2)` where `w = 1/(1 + β·M)`, `M` is per-image-normalized Gaussian-blurred luma Sobel-gradient magnitude of the cover, β=4.0. FFL term unchanged. All other design003 settings preserved.

**Parent:** runs/idea001/design003

**Starting Point:** runs/idea001/design003

---

## 1. Motivation and approach

This is direction (1) of `runs/idea003/idea.md`. design003 lifted PSNR from 27.33 → 27.77 by reweighting the host loss across **frequency/chroma channels** (YUV+FFL, no LPIPS). It still applies the YUV penalty *spatially uniformly* — a smooth sky pixel and a tree-canopy pixel are penalized identically, even though human vision tolerates ~10× more residual energy in textured regions (classical JND / spread-spectrum-in-textured-regions intuition).

This design adds a **per-pixel content-adaptive weighting** to the YUV term only:
- `w(cover) = 1 / (1 + β · M(cover))` where `M(cover) ∈ [0,1]` is the per-image-normalized Gaussian-blurred Sobel gradient magnitude of the cover's luma channel.
- `M` is computed under `torch.no_grad()` from `cover` only and detached; it is a *weight*, not a gradient source. The encoder cannot game the mask because there is no gradient path through `M`.
- The FFL term is unchanged (FFL is a global frequency statistic; per-pixel spatial weighting is ill-defined for a frequency-domain loss).
- All design003 hyperparameters (`yuv_w=1.5`, `ffl_w=1.5`, `lpips_w=0.0`, `secret_weight=20.0`, `--num_blocks 16`, `--subnet_type rdb`, `--lr 2e-4`, `--stage_a_steps 5000`, `--stage_b_steps 10000`) are preserved.

Rationale for β=4.0: in [0,1]-normalized `M`, the maximally-textured pixel gets weight `1/(1+4)=0.20` and a flat pixel gets weight `1/(1+0)=1.00` — a 5× dynamic range. This is the conservative end of the idea's stated `≤4.0` ceiling, chosen to preserve `bit_acc_clean=1.000` (idea constraint).

## 2. Files changed (Builder may only edit these)

- `code/baseline/losses.py` — add JND mask helper functions and apply per-pixel weighting inside `ImageSecretLoss.forward` on the YUV term only. Add a `jnd_beta` constructor argument (default `0.0` = off; this design sets `4.0`).
- `code/baseline/exp0_inn_train.py` — pass `jnd_beta=4.0` to `ImageSecretLoss(...)` constructor; add `l_yuv_unweighted` (and `jnd_beta`) to `metrics.jsonl` per-step record for diagnostics.

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

No new CLI flags are added; `jnd_beta` is hard-coded inside the `ImageSecretLoss(...)` instantiation in `exp0_inn_train.py`.

## 4. Algorithmic changes (concrete)

### 4.1 `code/baseline/losses.py` — add JND mask + weighted YUV MSE

**Add module-level helpers** near the top of `losses.py` (after the `_RGB2YUV` / `rgb_to_yuv` definitions):

```python
# Sobel kernels (3x3) for luma-channel gradient magnitude.
_SOBEL_X = torch.tensor([[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.],
                         [ 0.,  0.,  0.],
                         [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)

def _gaussian_kernel_5x5(device, dtype):
    # Symmetric 5x5 Gaussian (sigma≈1), normalized to sum=1.
    k = torch.tensor([1., 4., 6., 4., 1.], dtype=dtype, device=device)
    k2 = (k[:, None] * k[None, :])
    k2 = k2 / k2.sum()
    return k2.view(1, 1, 5, 5)

@torch.no_grad()
def sobel_jnd_mask(cover: torch.Tensor) -> torch.Tensor:
    """Per-image-normalized Gaussian-blurred Sobel gradient magnitude of the
    luma channel.  Input `cover` is in [-1, 1], shape (B, 3, H, W).  Output
    shape (B, 1, H, W), values in [0, 1] (per-image max-normalized).

    Returns a tensor detached from the autograd graph.
    """
    # Luma in [-1, 1] -> [0, 1] for stable gradient magnitudes.
    yuv = rgb_to_yuv(cover)
    luma = (yuv[:, 0:1] + 1.0) * 0.5  # (B,1,H,W) in [0,1]
    sx = _SOBEL_X.to(luma.device, luma.dtype)
    sy = _SOBEL_Y.to(luma.device, luma.dtype)
    gx = F.conv2d(luma, sx, padding=1)
    gy = F.conv2d(luma, sy, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)  # (B,1,H,W)
    # Gaussian blur (5x5, σ≈1) to make the mask smooth (no per-pixel speckle).
    g = _gaussian_kernel_5x5(luma.device, luma.dtype)
    mag = F.conv2d(mag, g, padding=2)
    # Per-image max-normalize to [0, 1]; clamp denominator for safety.
    B = mag.shape[0]
    flat_max = mag.view(B, -1).amax(dim=1).clamp_min(1e-6)
    mag = mag / flat_max.view(B, 1, 1, 1)
    return mag.detach()
```

**Modify `ImageSecretLoss.__init__`** to accept `jnd_beta`:

```python
def __init__(self,
             secret_weight: float = 20.0,
             yuv_w: float = 1.5,
             lpips_w: float = 1.0,
             ffl_w: float = 1.5,
             jnd_beta: float = 0.0):
    super().__init__()
    self.secret_weight = secret_weight
    self.yuv_w = yuv_w
    self.lpips_w = lpips_w
    self.ffl_w = ffl_w
    self.jnd_beta = jnd_beta
    self.lpips = lpips_lib.LPIPS(net="alex", verbose=False)
    for p in self.lpips.parameters():
        p.requires_grad_(False)
    self.ffl = FocalFrequencyLoss()
```

**Modify `ImageSecretLoss.forward`** to apply per-pixel JND weighting on the YUV term. Replace the existing line
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
    M = sobel_jnd_mask(cover)             # (B, 1, H, W) in [0, 1], detached
    w = 1.0 / (1.0 + self.jnd_beta * M)   # (B, 1, H, W), broadcast over 3 chans
    # mean over (B, 3, H, W) of w*sq, with w broadcast across the 3 YUV chans
    l_yuv = (w * sq).mean()
else:
    l_yuv = l_yuv_unweighted
```

Add `l_yuv_unweighted` to the returned `logs` dict alongside the existing fields:
```python
logs = dict(
    loss=total.detach(),
    l_bce=l_bce.detach(),
    l_yuv=l_yuv.detach(),
    l_yuv_unweighted=l_yuv_unweighted.detach(),
    l_lpips=l_lpips.detach(),
    l_ffl=l_ffl.detach(),
    l_quality=l_quality.detach(),
    alpha=alpha,
)
```

The LPIPS short-circuit (lines 61–65 of design003's `losses.py`) and the rest of `forward` are unchanged.

### 4.2 `code/baseline/exp0_inn_train.py` — pass `jnd_beta=4.0`

Modify the `ImageSecretLoss(...)` call (currently at lines 258–263 of design003's file):
```python
img_loss_mod = ImageSecretLoss(
    secret_weight=args.bit_weight,   # 20.0
    yuv_w=1.5,
    lpips_w=0.0,
    ffl_w=1.5,
    jnd_beta=4.0,
).to(device)
```

In the per-step `metrics.jsonl` record (the `rec = dict(...)` block around design003 line 439), add:
```python
l_yuv_unweighted=float(loss_logs["l_yuv_unweighted"]),
jnd_beta=4.0,
```
alongside the existing `l_yuv`, `l_lpips`, `l_ffl`, `l_quality`, `quality_alpha` fields. (`l_yuv` itself becomes the *weighted* mean; `l_yuv_unweighted` lets us read the raw MSE for comparison against design003.)

No other change to `exp0_inn_train.py`. Optimizer, AMP, clipping, curriculum, validation, checkpointing all unchanged.

## 5. Expected behavior after the change

- `metrics.jsonl` per-step records gain two new fields: `l_yuv_unweighted` (float, ≥ `l_yuv`) and `jnd_beta` (4.0).
- `l_yuv` (weighted) at convergence should be roughly 30–60% smaller than `l_yuv_unweighted`, since most pixels in natural images are flat (M small → w near 1) and a minority are textured (M large → w as low as 0.2). The exact ratio depends on the dataset's texture statistics.
- `bit_acc_clean = 1.000` must hold at every eval resolution (idea constraint; β=4.0 chosen conservatively to preserve secret pathway).
- Hypothesis on outcome: PSNR (and `score`) at step 30000 ≥ design003 (27.77). A modest +0.1–0.5 dB gain is the success target; flat or worse means texture-energy redistribution alone is insufficient and Sobel-edges are not the right mask geometry (motivating design002's variance-based mask).
- Wallclock per step: design003 + ~1 ms (Sobel + Gaussian + max-normalize on 16×256² fp32 ≈ <1 ms on a 2080Ti). Total walltime within design003's headroom.

## 6. Constraints and invariants Builder must preserve

- No edits to `infra/**`. No edits to `code/baseline/inn_model.py` or `code/baseline/train.py`.
- Locked CLI flags untouched (`--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`).
- All non-locked CLI flags identical to design003 parent (`--num_blocks 16`, `--subnet_type rdb`, `--lr 2e-4`, `--img_weight 1.0`, `--bit_weight 20.0`, `--stage_a_steps 5000`, `--stage_b_steps 10000`).
- LPIPS remains off (`lpips_w=0.0`); short-circuit preserved.
- `secret_weight=20.0` preserved.
- JND mask is computed under `torch.no_grad()` from `cover` only and detached. Builder MUST NOT compute it from `stego`, MUST NOT pass it through any module that records autograd, and MUST NOT make `M` a function of model parameters.
- Per-image normalization (`/ flat_max`) MUST be applied so β has consistent semantics across covers regardless of overall texture level.
- Optimizer, scheduler, AMP autocast scope, gradient clipping, curriculum α-ramp, validation cadence, checkpoint format unchanged.
- The hard condition `bit_acc_clean = 1.000` must remain achievable at convergence. If during local smoke testing the Builder observes `bit_acc_clean` regressing below 1.0 at the `--max_steps 30000` checkpoint, this is a Designer-level concern — flag to Orchestrator, do not silently increase `--bit_weight`.

## 7. Edge cases

- **Constant-luma cover** (all-gray image): `mag` is all zeros → `flat_max` clamps to 1e-6 → `M` ≈ 0 everywhere → `w = 1` everywhere → loss reduces to design003's unweighted YUV MSE. Correct degenerate behavior.
- **Single-pixel hot edge**: per-image max-normalize sets that pixel's `M` to 1.0 and everywhere else proportionally smaller; Gaussian blur (5×5) before normalization smooths the speckle so the hot pixel is not numerically isolated.
- **AMP / autocast**: `sobel_jnd_mask` is called inside the existing fp32-wrapped `with torch.amp.autocast("cuda", enabled=False):` block (since `cover_fp32` is the input to `img_loss_mod`), so kernels run in fp32. No fp16 numerical issues on the `sqrt`.
- **Channel broadcast**: `w` has shape `(B, 1, H, W)` and `sq` has shape `(B, 3, H, W)`; PyTorch broadcasts `w` across the 3 YUV channels — i.e., the same per-pixel weight is applied to Y, U, V. This is intentional: the JND mask is a luma-derived spatial prior and is not chroma-specific.
- **Padding mode**: `F.conv2d(..., padding=1)` for Sobel and `padding=2` for the 5×5 Gaussian use zero-padding; this slightly under-estimates `M` on the 1–2-pixel image border. Acceptable: borders are negligible relative to the 256² interior, and per-image max-normalization makes the absolute scale irrelevant.
- **fp16 dtype propagation**: Sobel/Gaussian kernels are constructed with `dtype=torch.float32` at module-load and `.to(luma.device, luma.dtype)` at call time, so they match `cover_fp32`'s dtype. No silent dtype upcast.
