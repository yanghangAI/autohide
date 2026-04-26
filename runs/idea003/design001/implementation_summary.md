**Files changed:**
code/losses.py
code/exp0_inn_train.py

**Changes:**

- `code/losses.py` — added Sobel kernels, a 5x5 Gaussian helper, and a `sobel_jnd_mask(cover)` function that returns a per-image-normalized, Gaussian-blurred Sobel-gradient magnitude of the luma channel under `torch.no_grad()`. Added a `jnd_beta` argument to `ImageSecretLoss.__init__` (default 0.0). Replaced the unweighted YUV MSE with a per-pixel JND-weighted MSE `(w * sq).mean()` where `w = 1/(1 + jnd_beta · M)`, falling back to the unweighted mean when `jnd_beta == 0`. Added `l_yuv_unweighted` to the returned `logs` dict for diagnostics.
- `code/exp0_inn_train.py` — passed `jnd_beta=4.0` to the `ImageSecretLoss(...)` constructor and added `l_yuv_unweighted` and `jnd_beta=4.0` to the per-step `metrics.jsonl` record. No CLI flag changes.

In `code/losses.py`, the helpers:

```python
# Sobel kernels (3x3) for luma-channel gradient magnitude.
_SOBEL_X = torch.tensor([[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1., -2., -1.],
                         [ 0.,  0.,  0.],
                         [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
```

In `code/losses.py`, the mask:

```python
@torch.no_grad()
def sobel_jnd_mask(cover: torch.Tensor) -> torch.Tensor:
    """Per-image-normalized Gaussian-blurred Sobel gradient magnitude of the
    luma channel.  Input `cover` is in [-1, 1], shape (B, 3, H, W).  Output
    shape (B, 1, H, W), values in [0, 1] (per-image max-normalized).

    Returns a tensor detached from the autograd graph.
    """
    yuv = rgb_to_yuv(cover)
    luma = (yuv[:, 0:1] + 1.0) * 0.5  # (B,1,H,W) in [0,1]
    sx = _SOBEL_X.to(luma.device, luma.dtype)
    sy = _SOBEL_Y.to(luma.device, luma.dtype)
    gx = F.conv2d(luma, sx, padding=1)
    gy = F.conv2d(luma, sy, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)  # (B,1,H,W)
    g = _gaussian_kernel_5x5(luma.device, luma.dtype)
    mag = F.conv2d(mag, g, padding=2)
    B = mag.shape[0]
    flat_max = mag.view(B, -1).amax(dim=1).clamp_min(1e-6)
    mag = mag / flat_max.view(B, 1, 1, 1)
    return mag.detach()
```

In `code/losses.py`, the weighted YUV term:

```python
        # Quality components
        yuv_stego = rgb_to_yuv(stego)
        yuv_cover = rgb_to_yuv(cover)
        sq = (yuv_stego - yuv_cover) ** 2  # (B, 3, H, W)
        l_yuv_unweighted = sq.mean()
        if self.jnd_beta > 0.0:
            M = sobel_jnd_mask(cover)             # (B, 1, H, W) in [0, 1], detached
            w = 1.0 / (1.0 + self.jnd_beta * M)   # (B, 1, H, W), broadcast over 3 chans
            l_yuv = (w * sq).mean()
        else:
            l_yuv = l_yuv_unweighted
```

In `code/exp0_inn_train.py`, the constructor call:

```python
    img_loss_mod = ImageSecretLoss(
        secret_weight=args.bit_weight,   # 20.0
        yuv_w=1.5,
        lpips_w=0.0,
        ffl_w=1.5,
        jnd_beta=4.0,
    ).to(device)
```

In `code/exp0_inn_train.py`, the per-step record:

```python
                       l_yuv=loss_logs["l_yuv"].item(),
                       l_yuv_unweighted=float(loss_logs["l_yuv_unweighted"]),
                       jnd_beta=4.0,
                       l_lpips=loss_logs["l_lpips"].item(),
```
