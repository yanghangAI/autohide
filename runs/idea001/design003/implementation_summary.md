**Files changed:**
code/exp0_inn_train.py
code/losses.py

**Changes:**
- `code/losses.py`: Added a `self.lpips_w == 0.0` short-circuit guard in `ImageSecretLoss.forward` so the LPIPS AlexNet forward is skipped (and `l_lpips` is set to a zero scalar with the right device/dtype) when the LPIPS weight is zero. The `self.lpips` module construction in `__init__` is preserved unchanged.
- `code/exp0_inn_train.py`: Imported `ImageSecretLoss` from `losses`, instantiated it after `INNCodec` construction with the YUV+FFL-only config (`yuv_w=1.5, lpips_w=0.0, ffl_w=1.5, secret_weight=args.bit_weight`), replaced the inline L1 + BCE block with a single `img_loss_mod(...)` call wrapped in `autocast(enabled=False)` for fp32 stability, and extended the per-step `metrics.jsonl` record with `l_yuv`, `l_lpips` (always ~0), `l_ffl`, `l_quality`, and `quality_alpha`.

In `code/losses.py`, the LPIPS short-circuit:

```python
        if self.lpips_w == 0.0:
            # Skip the AlexNet forward when the weight is zero — saves ~5-10 ms/step.
            l_lpips = torch.zeros((), device=stego.device, dtype=stego.dtype)
        else:
            l_lpips = self.lpips(stego, cover).mean()  # inputs already in [-1, 1]
```

In `code/exp0_inn_train.py`, the new import:

```python
from inn_model import INNCodec  # noqa: E402  (sibling, baseline-owned)
from losses import ImageSecretLoss  # noqa: E402  (sibling, baseline-owned)
```

In `code/exp0_inn_train.py`, the loss-module instantiation right after the `INNCodec` construction:

```python
    img_loss_mod = ImageSecretLoss(
        secret_weight=args.bit_weight,   # 20.0
        yuv_w=1.5,
        lpips_w=0.0,
        ffl_w=1.5,
    ).to(device)
```

In `code/exp0_inn_train.py`, the new train-step loss block replacing the L1+BCE inline:

```python
            # Quality alpha tied to attack curriculum: clean phase = 0
            # (pure bit pressure), full = full quality term.
            quality_alpha = args.img_weight * attack_alpha
            container_fp32 = container.float()
            cover_fp32 = cover.float()
            logits_fp32 = logits.float()
            with torch.amp.autocast("cuda", enabled=False):
                loss, loss_logs = img_loss_mod(
                    cover=cover_fp32,
                    stego=container_fp32,
                    secret=bits,
                    logits=logits_fp32,
                    alpha=quality_alpha,
                )
            bit_loss = loss_logs["l_bce"]
            img_loss = loss_logs["l_quality"]   # backward-compatible logging
```

In `code/exp0_inn_train.py`, the extended logging record:

```python
                       l_yuv=loss_logs["l_yuv"].item(),
                       l_lpips=loss_logs["l_lpips"].item(),
                       l_ffl=loss_logs["l_ffl"].item(),
                       l_quality=loss_logs["l_quality"].item(),
                       quality_alpha=quality_alpha,
```
