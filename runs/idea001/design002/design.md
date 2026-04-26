**Design Description:** LPIPS-dominant perceptual rebalancing — `ImageSecretLoss` with `lpips_w=3.0`, shrunken `yuv_w=0.5`, `ffl_w=0.5`, gated by curriculum α-ramp; tests whether perceptual-only quality drives PSNR via residual reshaping.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Motivation and approach

This design tests the hypothesis from `idea.md` direction (2): if perceptual-only quality (LPIPS, which does not directly reward MSE/PSNR) still raises measured PSNR — by pushing residual energy into perceptually-tolerated bands the INN otherwise wouldn't favour — then LPIPS is the dominant useful term. Compared to design001 (balanced default weights), this design tilts the quality bundle toward LPIPS (×3) and damps the YUV/FFL bands (×1/3 each). Bit pressure is unchanged from baseline (`secret_weight=20.0`).

## 2. Files changed (Builder may only edit these)

- `code/baseline/exp0_inn_train.py` — wire `ImageSecretLoss` into the train loop with the LPIPS-dominant config below.
- `code/baseline/losses.py` — unchanged; reused as-is.

Untouched (locked): everything in `infra/**`, `code/baseline/inn_model.py`.

## 3. CLI flag values for `baseline/exp0_inn_train.py` (config)

Locked (must NOT change): `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.

Tuned for this design:
- `--num_blocks 16`
- `--subnet_type rdb`
- `--lr 2e-4`
- `--img_weight 1.0` — multiplier on `l_quality` (passed as part of `quality_alpha`).
- `--bit_weight 20.0` — passed as `secret_weight` to `ImageSecretLoss`.
- `--stage_a_steps 5000`
- `--stage_b_steps 10000`

All other flags keep their baseline defaults.

## 4. Algorithmic changes (concrete)

### 4.1 Instantiate the loss module

In `code/baseline/exp0_inn_train.py`, after `model = INNCodec(...)`:
```python
from losses import ImageSecretLoss
img_loss_mod = ImageSecretLoss(
    secret_weight=args.bit_weight,   # 20.0
    yuv_w=0.5,
    lpips_w=3.0,
    ffl_w=0.5,
).to(device)
```

### 4.2 Replace the inline loss computation

Inside the `autocast` block, replace the L1+BCE block exactly as in design001 §4.2/§4.3, including the fp32 wrapper around the loss call:
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

The single `img_loss_mod` call evaluates `secret_weight * BCE + alpha * (yuv_w * L_YUV + lpips_w * L_LPIPS + ffl_w * L_FFL)`. Do NOT add `args.bit_weight * bit_loss` again.

### 4.3 Logging additions

Same as design001 §4.4: extend `metrics.jsonl` per-step record with `l_yuv`, `l_lpips`, `l_ffl`, `l_quality`, `quality_alpha`. Existing fields preserved.

## 5. Expected behavior after the change

- `metrics.jsonl`: new fields above; `l_lpips` is the dominant of the three quality components.
- `bit_acc_clean = 1.000` at every eval resolution must hold (same `secret_weight=20.0` as baseline).
- Hypothesis: `psnr_ema` at step 30000 is ≥ baseline despite LPIPS not directly rewarding PSNR; if `psnr_ema` is *worse* than design001, the result is informative — it means the YUV+FFL terms (frequency/chroma reweighting) carry the PSNR gain rather than perceptual structure.
- `score = 2 * psnr_mean * (bit_acc_mean - 0.5)` recorded in `metrics.csv` after eval.
- Wallclock: same as design001 (LPIPS-AlexNet dominates the added cost; FFL/YUV are cheap).

## 6. Constraints and invariants Builder must preserve

Identical to design001 §6:
- No edits to `infra/**`.
- Locked CLI flags untouched.
- `secret_weight=20.0` (= `--bit_weight`) preserved to keep `bit_acc_clean=1.000` achievable.
- Optimizer / scheduler / AMP / clip / resume logic unchanged.
- Curriculum semantics preserved: `quality_alpha == 0` during clean phase.
- All edits inside `code/baseline/`.

## 7. Edge cases

- Identical to design001 §7. The only difference between this design and design001 is the three weight constants in the `ImageSecretLoss(...)` constructor.
