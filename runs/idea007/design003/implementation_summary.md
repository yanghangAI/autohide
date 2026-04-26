**Files changed:**
code/losses.py
code/exp0_inn_train.py

**Changes:**
- `code/losses.py` — Combined the design001 and design002 edits. Added `_gaussian_kernel_1d`, `_gaussian_blur_luma`, `multiscale_sobel_jnd_mask` (multi-scale Sobel JND mask) and `_sobel_mag_luma` (differentiable luma-Sobel helper for the residual). Extended `ImageSecretLoss.__init__` with `jnd_sigmas`, `jnd_pct`, `tv_w`, `tv_eps`, and accepted `jnd_mode="sobel_multiscale"`. Inside `forward`, dispatches to the multi-scale mask helper and adds the complementary TV term reusing the same `w`. Added `tv_w * l_tv` to `l_quality` and `l_tv` to the logs.
- `code/exp0_inn_train.py` — Passed `jnd_mode="sobel_multiscale"`, `jnd_sigmas=(1.0, 2.0, 4.0)`, `jnd_pct=0.99`, and `tv_w=0.3` to the `ImageSecretLoss(...)` call. Updated the per-step `metrics.jsonl` record to log the new mask + TV descriptor fields.

In `code/losses.py`, the multi-scale mask helper:

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

In `code/losses.py`, the differentiable luma-Sobel helper for the residual:

```python
def _sobel_mag_luma(rgb: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of the luma channel of an RGB tensor.
    Input shape (B, 3, H, W); output shape (B, 1, H, W). NOT detached —
    suitable for use inside a differentiable loss term on the residual.
    """
```

In `code/losses.py`, the combined constructor signature:

```python
                 jnd_mode: str = "sobel",
                 jnd_sigmas: tuple[float, ...] = (1.0, 2.0, 4.0),
                 jnd_pct: float = 0.99,
                 tv_w: float = 0.0,
                 tv_eps: float = 1e-6):
        super().__init__()
        assert jnd_mode in ("sobel", "variance", "sobel_multiscale"), f"unknown jnd_mode={jnd_mode}"
```

In `code/losses.py`, the dispatch branch in `forward`:

```python
            elif self.jnd_mode == "sobel_multiscale":
                M = multiscale_sobel_jnd_mask(cover, sigmas=self.jnd_sigmas, pct=self.jnd_pct)
```

In `code/losses.py`, the TV term reusing the multi-scale `w`:

```python
            if self.tv_w > 0.0:
                # Per-image normalize w to [0, 1] for the complementary mask.
                B = w.shape[0]
                w_max = w.view(B, -1).amax(dim=1).clamp_min(self.tv_eps)
                w_norm = w / w_max.view(B, 1, 1, 1)        # (B, 1, H, W) in [0, 1]
                comp = 1.0 - w_norm                         # 1 in flat regions, 0 in max-textured
                # Sobel magnitude of luma residual; differentiable wrt stego.
                g_res = _sobel_mag_luma(stego - cover)      # (B, 1, H, W)
                l_tv = (comp * g_res).mean()
```

In `code/losses.py`, the updated `l_quality`:

```python
        l_quality = (self.yuv_w * l_yuv
                     + self.lpips_w * l_lpips
                     + self.ffl_w * l_ffl
                     + self.pixel_w * l_pixel
                     + self.tv_w * l_tv)
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
        tv_w=0.3,
    ).to(device)
```

In `code/exp0_inn_train.py`, the new metrics record fields:

```python
                       jnd_mode="sobel_multiscale",
                       jnd_sigmas=[1.0, 2.0, 4.0],
                       jnd_pct=0.99,
                       l_tv=float(loss_logs["l_tv"]),
                       tv_w=0.3,
```
