**Files changed:**
code/losses.py
code/exp0_inn_train.py

**Changes:**

- `code/losses.py` — added Sobel kernels, the 5x5 Gaussian helper, the `sobel_jnd_mask(cover)` function (same as design001), and a `variance_jnd_mask(cover, win=7)` helper for symmetry. Added `jnd_beta`, `pixel_w`, and `jnd_mode` arguments to `ImageSecretLoss.__init__` (default `0.0` / `0.0` / `"sobel"`). The forward pass now computes the JND mask once and reuses it for both the weighted YUV-MSE term and a new masked L1 pixel-residual term `(w * |stego - cover|).mean()`. The mask is computed under `torch.no_grad()` and detached. The `l_quality` accumulation includes `self.pixel_w * l_pixel`. Added `l_yuv_unweighted` and `l_pixel` to the returned `logs` dict.
- `code/exp0_inn_train.py` — passed `jnd_beta=4.0, pixel_w=0.5, jnd_mode="sobel"` to `ImageSecretLoss(...)`; added `l_yuv_unweighted`, `l_pixel`, `jnd_beta=4.0`, `pixel_w=0.5`, `jnd_mode="sobel"` to the per-step `metrics.jsonl` record.

In `code/losses.py`, the constructor:

```python
    def __init__(self,
                 secret_weight: float = 20.0,
                 yuv_w: float = 1.5,
                 lpips_w: float = 1.0,
                 ffl_w: float = 1.5,
                 jnd_beta: float = 0.0,
                 pixel_w: float = 0.0,
                 jnd_mode: str = "sobel"):
        super().__init__()
        assert jnd_mode in ("sobel", "variance"), f"unknown jnd_mode={jnd_mode}"
```

In `code/losses.py`, the JND mask reuse and L1 pixel term:

```python
        if self.jnd_beta > 0.0:
            if self.jnd_mode == "sobel":
                M = sobel_jnd_mask(cover)
            else:
                M = variance_jnd_mask(cover, win=7)
            w = 1.0 / (1.0 + self.jnd_beta * M)      # (B, 1, H, W), broadcast over chans
            l_yuv = (w * sq).mean()
            if self.pixel_w > 0.0:
                # Reuse the same mask w; broadcast over RGB channels.
                l_pixel = (w * (stego - cover).abs()).mean()
            else:
                l_pixel = torch.zeros((), device=stego.device, dtype=stego.dtype)
```

In `code/losses.py`, the `l_quality` accumulation:

```python
        l_quality = (self.yuv_w * l_yuv
                     + self.lpips_w * l_lpips
                     + self.ffl_w * l_ffl
                     + self.pixel_w * l_pixel)
```

In `code/losses.py`, the logs dict:

```python
        logs = dict(
            loss=total.detach(),
            l_bce=l_bce.detach(),
            l_yuv=l_yuv.detach(),
            l_yuv_unweighted=l_yuv_unweighted.detach(),
            l_lpips=l_lpips.detach(),
            l_ffl=l_ffl.detach(),
            l_pixel=l_pixel.detach(),
            l_quality=l_quality.detach(),
            alpha=alpha,
        )
```

In `code/exp0_inn_train.py`, the constructor call:

```python
    img_loss_mod = ImageSecretLoss(
        secret_weight=args.bit_weight,   # 20.0
        yuv_w=1.5,
        lpips_w=0.0,
        ffl_w=1.5,
        jnd_beta=4.0,
        pixel_w=0.5,
        jnd_mode="sobel",
    ).to(device)
```

In `code/exp0_inn_train.py`, the per-step record:

```python
                       l_yuv=loss_logs["l_yuv"].item(),
                       l_yuv_unweighted=float(loss_logs["l_yuv_unweighted"]),
                       l_pixel=float(loss_logs["l_pixel"]),
                       jnd_beta=4.0,
                       pixel_w=0.5,
                       jnd_mode="sobel",
                       l_lpips=loss_logs["l_lpips"].item(),
```
