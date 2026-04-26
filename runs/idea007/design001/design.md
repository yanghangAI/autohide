**Design Description:** Multi-scale Sobel JND mask — replace the parent's single-scale Sobel-blur mask with a per-pixel max over Sobel-magnitudes computed on luma at three Gaussian blur scales (σ ∈ {1.0, 2.0, 4.0} px), each per-image normalized by its 99th percentile before aggregation. Use the resulting mask `w_multi = 1/(1 + β·M_multi)` (β=4.0) verbatim wherever the parent uses its single-scale `w` (YUV term and masked L1 term). All other parent settings preserved.

**Parent:** runs/idea003/design003

**Starting Point:** runs/idea003/design003

---

## 1. Goal and motivation

This is direction (1) of `runs/idea007/idea.md`. The parent (`runs/idea003/design003`, score=28.16, current best `Done`) uses a single-scale Sobel + 5×5-Gaussian-blur mask on luma. A single Sobel kernel detects structure at one effective scale only — fine textures, mid-frequency wood grain, and large architectural edges all register as "high gradient" but have very different perceptual masking profiles, and a single-scale mask cannot distinguish them. This design swaps the single-scale `sobel_jnd_mask(cover)` for a multi-scale variant that computes Sobel magnitudes on three Gaussian-blurred copies of the cover at σ ∈ {1.0, 2.0, 4.0} px, normalizes each per-image by its 99th percentile (outlier-robust), and takes the elementwise max across scales.

This isolates the mask-quality axis: every other knob — INN architecture, optimizer, loss weights, curriculum, `pixel_w=0.5`, `secret_weight=20.0`, `jnd_beta=4.0` — is identical to the parent. The only difference vs the parent is the function that produces `M`.

If `score > 28.16` (parent), broadening mask scale alone provides headroom. If equal, single-scale is already saturating the mask-shape signal at fixed β. If lower, multi-scale max-aggregation widens the "high-gradient" set too aggressively (over-allowing residuals in mid-scale textures that are actually visible) and we should retreat to single-scale.

## 2. Files changed (Builder may only edit these)

Builder may only touch the following files inside `code/` (mirroring `baseline/`):

- `code/losses.py` — add `multiscale_sobel_jnd_mask(cover, sigmas=(1.0, 2.0, 4.0), pct=0.99)` helper; add a new `jnd_mode` value `"sobel_multiscale"`; route the mask compute through it when `jnd_mode == "sobel_multiscale"`. No other algorithmic change to the loss.
- `code/exp0_inn_train.py` — change the `ImageSecretLoss(...)` call so `jnd_mode="sobel_multiscale"`; emit `jnd_mode="sobel_multiscale"` in the per-step `metrics.jsonl` record.

Untouched (locked or out of scope for this design):
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

No new CLI flags. The mask configuration (`jnd_beta=4.0`, `pixel_w=0.5`, `jnd_mode="sobel_multiscale"`, `sigmas=(1.0, 2.0, 4.0)`, `pct=0.99`) is hard-coded in the `ImageSecretLoss(...)` constructor call inside `code/exp0_inn_train.py` (mirroring how the parent hard-codes `jnd_mode="sobel"`).

## 4. Algorithmic spec (concrete)

### 4.1 `code/losses.py`

**Keep** the existing `_RGB2YUV`, `rgb_to_yuv`, `_SOBEL_X`, `_SOBEL_Y`, `_gaussian_kernel_5x5`, `sobel_jnd_mask`, `variance_jnd_mask`, and the `ImageSecretLoss` class exactly as in the parent. The parent's single-scale `sobel_jnd_mask` and `variance_jnd_mask` MUST remain, so existing `jnd_mode="sobel"` / `"variance"` paths still work; this design only adds a new branch.

**Add** a parametric Gaussian kernel builder (the parent only ships a fixed σ≈1 5×5 kernel; multi-scale needs σ ∈ {1, 2, 4}):

```python
def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    """1-D Gaussian kernel; radius = ceil(3*sigma), normalized to sum=1."""
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k  # shape (2*radius+1,)


def _gaussian_blur_luma(luma: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 2-D Gaussian blur on a single-channel image, padding='replicate'.
    `luma` shape (B, 1, H, W)."""
    k = _gaussian_kernel_1d(sigma, luma.device, luma.dtype)
    radius = (k.numel() - 1) // 2
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    x = F.pad(luma, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    x = F.conv2d(x, ky)
    return x
```

**Add** the multi-scale mask helper:

```python
@torch.no_grad()
def multiscale_sobel_jnd_mask(cover: torch.Tensor,
                              sigmas: tuple[float, ...] = (1.0, 2.0, 4.0),
                              pct: float = 0.99) -> torch.Tensor:
    """Per-pixel max over Sobel magnitudes of luma after Gaussian blurs at the
    specified sigmas. Each scale's magnitude map is normalized per-image by its
    99th percentile (clamped to >=1e-6) before aggregation, so beta has
    consistent semantics across covers and across scales.

    Input `cover` is in [-1, 1], shape (B, 3, H, W). Output shape (B, 1, H, W),
    values in [0, ~1] (a small fraction of pixels can exceed 1 because of the
    99th-percentile normalization; that is intentional and matches the parent's
    use of max-normalization being analogous). Detached from autograd graph.
    """
    yuv = rgb_to_yuv(cover)
    luma = (yuv[:, 0:1] + 1.0) * 0.5  # (B, 1, H, W) in [0, 1]
    sx = _SOBEL_X.to(luma.device, luma.dtype)
    sy = _SOBEL_Y.to(luma.device, luma.dtype)
    B = luma.shape[0]
    mags = []
    for sigma in sigmas:
        blurred = _gaussian_blur_luma(luma, sigma)
        gx = F.conv2d(blurred, sx, padding=1)
        gy = F.conv2d(blurred, sy, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)  # (B, 1, H, W)
        # Per-image 99th-percentile normalization
        flat = mag.view(B, -1)
        # torch.quantile is exact for float32 on the GPU
        q = torch.quantile(flat.float(), pct, dim=1).to(mag.dtype).clamp_min(1e-6)
        mag = mag / q.view(B, 1, 1, 1)
        mags.append(mag)
    M = torch.stack(mags, dim=0).amax(dim=0)  # (B, 1, H, W)
    return M.detach()
```

**Modify `ImageSecretLoss.__init__`** to accept `"sobel_multiscale"` in the `jnd_mode` assert and to store optional `sigmas` / `pct` kwargs:

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
             jnd_pct: float = 0.99):
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
    self.lpips = lpips_lib.LPIPS(net="alex", verbose=False)
    for p in self.lpips.parameters():
        p.requires_grad_(False)
    self.ffl = FocalFrequencyLoss()
```

**Modify `ImageSecretLoss.forward`** mask dispatch (only the `if self.jnd_beta > 0.0:` branch's mask compute changes):

```python
if self.jnd_beta > 0.0:
    if self.jnd_mode == "sobel":
        M = sobel_jnd_mask(cover)
    elif self.jnd_mode == "variance":
        M = variance_jnd_mask(cover, win=7)
    elif self.jnd_mode == "sobel_multiscale":
        M = multiscale_sobel_jnd_mask(cover, sigmas=self.jnd_sigmas, pct=self.jnd_pct)
    else:
        raise ValueError(f"unknown jnd_mode={self.jnd_mode}")
    w = 1.0 / (1.0 + self.jnd_beta * M)
    l_yuv = (w * sq).mean()
    if self.pixel_w > 0.0:
        l_pixel = (w * (stego - cover).abs()).mean()
    else:
        l_pixel = torch.zeros((), device=stego.device, dtype=stego.dtype)
else:
    # unchanged from parent
    ...
```

Everything else in `forward` (l_lpips short-circuit, l_ffl, l_quality formula, total, logs) is unchanged. The `logs` dict keys are unchanged.

### 4.2 `code/exp0_inn_train.py`

Modify the `ImageSecretLoss(...)` constructor call:

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
).to(device)
```

In the per-step `metrics.jsonl` record, change `jnd_mode` to `"sobel_multiscale"` and add the two new descriptor fields:

```python
jnd_mode="sobel_multiscale",
jnd_sigmas=[1.0, 2.0, 4.0],
jnd_pct=0.99,
```

(The per-step record continues to log `l_yuv`, `l_yuv_unweighted`, `l_pixel`, `jnd_beta=4.0`, `pixel_w=0.5` exactly as in the parent.)

No other change to `exp0_inn_train.py`.

## 5. Expected behavior after the change

- `metrics.jsonl` per-step records include `jnd_mode="sobel_multiscale"`, `jnd_sigmas=[1.0, 2.0, 4.0]`, `jnd_pct=0.99`, plus the existing `l_yuv`, `l_yuv_unweighted`, `l_pixel`, `jnd_beta=4.0`, `pixel_w=0.5`.
- `l_yuv` (masked) should be of similar order to the parent's, since `M_multi ∈ [0, ~1]` like the parent's `M` and β is unchanged. Expect `l_yuv` to drift slightly **lower** vs the parent at matched step (the multi-scale mask flags more "high gradient" pixels, so `w` is slightly lower across more pixels and the masked term is smaller numerically).
- `l_pixel` follows the same relationship.
- Hypothesis on outcome: PSNR ≥ parent (28.48); `score` ≥ 28.16; `bit_acc_clean_min` should not fall materially below the parent's 0.9949 (target ≥ 0.99). Bit_acc trades vs PSNR are bounded; a sharper map of where to put residuals should let the encoder spend its residual budget more efficiently, lifting PSNR without proportionally sacrificing bit_acc.
- Wallclock: three Gaussian blurs (radius 3, 6, 12) + three Sobel passes + three quantile-99 reductions, all on a (16, 1, 256, 256) tensor. <2 ms/step on a 2080Ti; total walltime ≤ parent + 1%.
- A design is `Done` when training reaches `step >= 30000`. After training, the locked multi-resolution eval (`infra/exp0_inn_eval.py`, 13 attacks × 9 resolutions) runs and writes `eval.json` → aggregated into `metrics.csv`. Primary metric: `score = 2 * psnr_mean * (bit_acc_mean - 0.5)`.

## 6. Constraints and invariants Builder must preserve

- No edits to `infra/**`. No edits to `code/inn_model.py` or `code/train.py`.
- Locked CLI flags untouched: `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`. The eval attack suite is locked.
- All non-locked CLI flags identical to the parent `runs/idea003/design003`: `--num_blocks 16`, `--subnet_type rdb`, `--lr 2e-4`, `--img_weight 1.0`, `--bit_weight 20.0`, `--stage_a_steps 5000`, `--stage_b_steps 10000`.
- `secret_weight=20.0`, `yuv_w=1.5`, `lpips_w=0.0`, `ffl_w=1.5`, `jnd_beta=4.0`, `pixel_w=0.5` preserved verbatim from parent. Only `jnd_mode` switches to `"sobel_multiscale"`.
- LPIPS off (`lpips_w=0.0`); the `if self.lpips_w == 0.0:` short-circuit is preserved.
- Mask MUST be computed inside `torch.no_grad()` (the helper's decorator handles this) and the returned tensor MUST be `.detach()`-ed. Mask is a weight, not a gradient source.
- Mask MUST be computed from `cover` only — never from `stego` and never from `secret`.
- The mask is computed exactly once per `forward` call and reused for both the YUV-weighted MSE and the masked-L1 pixel term.
- Per-image 99th-percentile normalization MUST be applied to each scale's magnitude map *before* the per-pixel max-aggregation across scales — never after. Aggregating then normalizing destroys the per-scale dynamic range and is forbidden.
- Existing `sobel_jnd_mask` and `variance_jnd_mask` helpers MUST NOT be deleted or modified — they are still referenced by `jnd_mode in ("sobel", "variance")` paths, which other designs depend on.
- Optimizer, LR scheduler, AMP autocast scope, gradient clipping, curriculum α-ramp, validation cadence, checkpoint format unchanged.
- `bit_acc_clean = 1.000` at every eval resolution remains the hard target. If a smoke run shows `bit_acc_clean_min < 0.99` at step 5000 onward, flag to Designer (would suggest β is too high under the multi-scale aggregation; do not silently retune).

## 7. Edge cases

- **Per-image 99th-percentile normalization vs max-normalization (parent)**: The parent uses `flat.amax(dim=1)`. We use `torch.quantile(flat, 0.99, dim=1)` to suppress single-pixel outliers from very sharp edges. A small fraction of pixels (≤ 1%) will have `M > 1` after this normalization; that is intentional. `w = 1/(1 + β·M)` remains in `(0, 1]` for `M ≥ 0`, so no numerical issue.
- **AMP / autocast**: `torch.quantile` is fp32-only on CUDA in older PyTorch builds. The helper casts `flat.float()` for the quantile compute and casts the result back to `mag.dtype`. The Sobel and Gaussian convs work in mixed precision (the parent already runs them under autocast).
- **Replicate padding for Gaussian blur**: Used for σ=4 (radius=12) where zero-padding would create dark borders that look like edges. Sobel uses `padding=1` (zero-pad), matching the parent.
- **Sigma scaling and image size**: At 256×256, σ=4 px (radius 12) is well within the image. No degenerate cases.
- **Constant-luma cover**: All three Sobel-mag maps are 0 → after dividing by `clamp_min(1e-6)` each map is 0 → `M_multi = 0` → `w = 1` → loss reduces to unmasked YUV-MSE + plain L1 (scaled by `yuv_w` and `pixel_w`). Well-defined; matches the parent's degenerate behavior.
- **`jnd_beta = 0` short-circuit**: Inherited from parent; the mask compute is skipped entirely. This design always sets `jnd_beta = 4.0`, so the short-circuit is not exercised, but it MUST remain in place for ablation use.
- **Mask shape broadcast**: `w` is `(B, 1, H, W)`; both `sq` `(B, 3, H, W)` and `(stego - cover).abs()` `(B, 3, H, W)` broadcast cleanly. Same as parent.
- **`metrics.jsonl` key stability**: `jnd_sigmas` is logged as a Python list `[1.0, 2.0, 4.0]` (JSON-serializable). Do not log the tuple directly — `json.dumps` cannot encode tuples without a custom encoder; the parent's logger uses default `json.dumps`.
