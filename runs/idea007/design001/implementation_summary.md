**Files changed:**
code/losses.py
code/exp0_inn_train.py

**Changes:**
- `code/losses.py` — Added two helper functions (`_gaussian_kernel_1d`, `_gaussian_blur_luma`) for parametric separable Gaussian blur and a new `multiscale_sobel_jnd_mask` helper that computes Sobel magnitudes on luma after Gaussian blurs at σ ∈ {1.0, 2.0, 4.0}, normalizes each per-image by its 99th percentile, and max-aggregates across scales. Extended `ImageSecretLoss.__init__` to accept `jnd_mode="sobel_multiscale"` plus `jnd_sigmas`/`jnd_pct` kwargs, and added the dispatch branch in `forward`.
- `code/exp0_inn_train.py` — Switched the `ImageSecretLoss(...)` constructor to `jnd_mode="sobel_multiscale"` with `jnd_sigmas=(1.0, 2.0, 4.0)` and `jnd_pct=0.99`. Updated the per-step `metrics.jsonl` record to log the new mask descriptor fields.

In `code/losses.py`, the new parametric Gaussian helpers:

```python
def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    """1-D Gaussian kernel; radius = ceil(3*sigma), normalized to sum=1."""
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k  # shape (2*radius+1,)
```

In `code/losses.py`, the multi-scale Sobel mask:

```python
@torch.no_grad()
def multiscale_sobel_jnd_mask(cover: torch.Tensor,
                              sigmas: tuple[float, ...] = (1.0, 2.0, 4.0),
                              pct: float = 0.99) -> torch.Tensor:
```

In `code/losses.py`, the per-scale normalize-then-aggregate body:

```python
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

In `code/losses.py`, the dispatch branch in `forward`:

```python
            if self.jnd_mode == "sobel":
                M = sobel_jnd_mask(cover)
            elif self.jnd_mode == "variance":
                M = variance_jnd_mask(cover, win=7)
            elif self.jnd_mode == "sobel_multiscale":
                M = multiscale_sobel_jnd_mask(cover, sigmas=self.jnd_sigmas, pct=self.jnd_pct)
            else:
                raise ValueError(f"unknown jnd_mode={self.jnd_mode}")
```

In `code/exp0_inn_train.py`, the new constructor call:

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

In `code/exp0_inn_train.py`, the updated metrics record fields:

```python
                       jnd_mode="sobel_multiscale",
                       jnd_sigmas=[1.0, 2.0, 4.0],
                       jnd_pct=0.99,
```
