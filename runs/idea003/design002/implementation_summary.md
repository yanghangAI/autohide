**Files changed:**
code/losses.py
code/exp0_inn_train.py

**Changes:**

- `code/losses.py` — added a `variance_jnd_mask(cover, win=7)` helper that computes the per-image-normalized local std-dev of the luma channel over a 7x7 uniform window using the `E[X^2] - E[X]^2` trick (with `clamp_min(0.0)` to guard against floating-point negatives) under `torch.no_grad()`. Added `jnd_beta` and `jnd_mode` arguments to `ImageSecretLoss.__init__` (default `0.0` / `"variance"`), with an assert restricting `jnd_mode` to `{"sobel", "variance"}`. Replaced the unweighted YUV MSE with a per-pixel JND-weighted MSE `(w * sq).mean()` where `w = 1/(1 + jnd_beta · M)`. Added `l_yuv_unweighted` to the returned `logs` dict.
- `code/exp0_inn_train.py` — passed `jnd_beta=4.0, jnd_mode="variance"` to the `ImageSecretLoss(...)` constructor and added `l_yuv_unweighted`, `jnd_beta=4.0`, and `jnd_mode="variance"` to the per-step `metrics.jsonl` record.

In `code/losses.py`, the variance mask:

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

In `code/losses.py`, the constructor:

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
```

In `code/losses.py`, the weighted YUV term:

```python
        # Quality components
        yuv_stego = rgb_to_yuv(stego)
        yuv_cover = rgb_to_yuv(cover)
        sq = (yuv_stego - yuv_cover) ** 2  # (B, 3, H, W)
        l_yuv_unweighted = sq.mean()
        if self.jnd_beta > 0.0:
            if self.jnd_mode == "variance":
                M = variance_jnd_mask(cover, win=7)   # (B, 1, H, W) in [0, 1], detached
            else:  # "sobel" — not exercised by this design
                M = variance_jnd_mask(cover, win=7)
            w = 1.0 / (1.0 + self.jnd_beta * M)       # (B, 1, H, W), broadcast over chans
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
        jnd_mode="variance",
    ).to(device)
```

In `code/exp0_inn_train.py`, the per-step record:

```python
                       l_yuv=loss_logs["l_yuv"].item(),
                       l_yuv_unweighted=float(loss_logs["l_yuv_unweighted"]),
                       jnd_beta=4.0,
                       jnd_mode="variance",
                       l_lpips=loss_logs["l_lpips"].item(),
```
