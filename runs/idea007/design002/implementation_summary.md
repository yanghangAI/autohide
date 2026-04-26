**Files changed:**
code/losses.py
code/exp0_inn_train.py

**Changes:**
- `code/losses.py` — Added `_sobel_mag_luma(rgb)` helper that returns the (differentiable) Sobel-magnitude of the luma channel, suitable for use on the residual `(stego - cover)`. Extended `ImageSecretLoss.__init__` with `tv_w` and `tv_eps` kwargs. Inside `forward`, after `w` and `l_pixel` are computed, added the complementary edge-aware TV term `l_tv = mean((1 - w_norm) · |∇(stego - cover)|_luma)` with `w_norm = w / w.amax(per-image)`. Added `tv_w * l_tv` to `l_quality` and `l_tv` to the logs dict.
- `code/exp0_inn_train.py` — Passed `tv_w=0.3` to the `ImageSecretLoss(...)` call. Added `l_tv` and `tv_w` to the per-step `metrics.jsonl` record.

In `code/losses.py`, the new differentiable luma-Sobel helper:

```python
def _sobel_mag_luma(rgb: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude of the luma channel of an RGB tensor.
    Input shape (B, 3, H, W); output shape (B, 1, H, W). NOT detached —
    suitable for use inside a differentiable loss term on the residual.
    """
    yuv = rgb_to_yuv(rgb)
    luma = yuv[:, 0:1]  # (B, 1, H, W)
    sx = _SOBEL_X.to(luma.device, luma.dtype)
    sy = _SOBEL_Y.to(luma.device, luma.dtype)
    gx = F.conv2d(luma, sx, padding=1)
    gy = F.conv2d(luma, sy, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag
```

In `code/losses.py`, the new constructor kwargs:

```python
                 jnd_mode: str = "sobel",
                 tv_w: float = 0.0,
                 tv_eps: float = 1e-6):
```

In `code/losses.py`, the TV term inside `forward`:

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
            else:
                l_tv = torch.zeros((), device=stego.device, dtype=stego.dtype)
```

In `code/losses.py`, the updated `l_quality` and `logs`:

```python
        l_quality = (self.yuv_w * l_yuv
                     + self.lpips_w * l_lpips
                     + self.ffl_w * l_ffl
                     + self.pixel_w * l_pixel
                     + self.tv_w * l_tv)
```

```python
            l_pixel=l_pixel.detach(),
            l_tv=l_tv.detach(),
            l_quality=l_quality.detach(),
```

In `code/exp0_inn_train.py`, the updated constructor call:

```python
    img_loss_mod = ImageSecretLoss(
        secret_weight=args.bit_weight,   # 20.0
        yuv_w=1.5,
        lpips_w=0.0,
        ffl_w=1.5,
        jnd_beta=4.0,
        pixel_w=0.5,
        jnd_mode="sobel",
        tv_w=0.3,
    ).to(device)
```

In `code/exp0_inn_train.py`, the new metrics record fields:

```python
                       jnd_mode="sobel",
                       l_tv=float(loss_logs["l_tv"]),
                       tv_w=0.3,
```
