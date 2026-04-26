**Design Description:** YUV+FFL only (no LPIPS) — chroma- and frequency-reweighted host loss with `yuv_w=1.5`, `ffl_w=1.5`, `lpips_w=0.0`; gated by curriculum α-ramp. Tests whether perceptual gains come from chroma/frequency reweighting alone, with no AlexNet forward.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Motivation and approach

This design tests `idea.md` direction (3): drop the LPIPS term entirely and rely on the YUV (luma/chroma) + FFL (focal frequency) perceptual reweighting only. If this design matches or beats design001, then the LPIPS-AlexNet forward is unnecessary cost; if it is materially worse, the deep-feature term is load-bearing. Bit pressure is unchanged.

Implementation note: `ImageSecretLoss` always instantiates an LPIPS module in `__init__` (see `baseline/losses.py:48`). To avoid the wasted forward when `lpips_w=0`, this design adds a lightweight short-circuit in `losses.py` so the LPIPS forward is skipped when `self.lpips_w == 0.0`.

## 2. Files changed (Builder may only edit these)

- `code/baseline/exp0_inn_train.py` — wire `ImageSecretLoss` into the train loop with the YUV+FFL-only config.
- `code/baseline/losses.py` — add a `self.lpips_w == 0.0` short-circuit in `ImageSecretLoss.forward` to skip the LPIPS forward; logged `l_lpips` becomes a zero scalar in this branch. No other behavior change.

Untouched (locked): everything in `infra/**`, `code/baseline/inn_model.py`.

## 3. CLI flag values for `baseline/exp0_inn_train.py` (config)

Locked (must NOT change): `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.

Tuned for this design:
- `--num_blocks 16`
- `--subnet_type rdb`
- `--lr 2e-4`
- `--img_weight 1.0` — multiplier on `l_quality`.
- `--bit_weight 20.0` — passed as `secret_weight`.
- `--stage_a_steps 5000`
- `--stage_b_steps 10000`

All other flags keep their baseline defaults.

## 4. Algorithmic changes (concrete)

### 4.1 `code/baseline/losses.py` — short-circuit LPIPS when weight is zero

Inside `ImageSecretLoss.forward`, replace:
```python
l_lpips = self.lpips(stego, cover).mean()  # inputs already in [-1, 1]
```
with:
```python
if self.lpips_w == 0.0:
    # Skip the AlexNet forward when the weight is zero — saves ~5-10 ms/step.
    l_lpips = torch.zeros((), device=stego.device, dtype=stego.dtype)
else:
    l_lpips = self.lpips(stego, cover).mean()  # inputs already in [-1, 1]
```
Everything else in `losses.py` is unchanged. (No removal of the `self.lpips = lpips_lib.LPIPS(...)` construction; that just sits idle.)

### 4.2 `code/baseline/exp0_inn_train.py` — instantiate and wire the loss

After `model = INNCodec(...)`:
```python
from losses import ImageSecretLoss
img_loss_mod = ImageSecretLoss(
    secret_weight=args.bit_weight,   # 20.0
    yuv_w=1.5,
    lpips_w=0.0,
    ffl_w=1.5,
).to(device)
```

Inside the training step `autocast` block, replace the inline L1+BCE exactly as in design001 §4.2/§4.3 with the fp32-wrapped call:
```python
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
img_loss = loss_logs["l_quality"]
```

Do NOT re-add `args.bit_weight * bit_loss`.

### 4.3 Logging additions

Same as design001 §4.4: `l_yuv`, `l_lpips` (will be ~0), `l_ffl`, `l_quality`, `quality_alpha` added to `metrics.jsonl` per-step record. Existing fields preserved.

## 5. Expected behavior after the change

- `metrics.jsonl` includes the new fields; `l_lpips` is exactly 0.0 throughout.
- No AlexNet forward each step → wallclock ~ baseline (within noise).
- `bit_acc_clean = 1.000` must hold at every eval resolution.
- `psnr_ema` at step 30000: hypothesis is between baseline and design001; informative either way.
- `score` recorded in `metrics.csv` after eval.

## 6. Constraints and invariants Builder must preserve

Identical to design001 §6:
- No edits to `infra/**`.
- Locked CLI flags untouched.
- `secret_weight=20.0` preserved.
- Optimizer / scheduler / AMP / clip / resume logic unchanged.
- Curriculum semantics preserved.
- All edits inside `code/baseline/`. The `losses.py` edit is a strict additive guard around the existing LPIPS call; do not delete the `self.lpips` construction (other designs reuse the module unchanged).

## 7. Edge cases

- The LPIPS module is still constructed in `__init__` and moved to device by `.to(device)`; this is intentional and harmless. Skipping is purely at the forward call.
- `l_lpips` zero-scalar uses `stego.dtype` so it composes cleanly with the fp32-wrapped loss math.
- All other edge cases identical to design001 §7.
