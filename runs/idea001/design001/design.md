**Design Description:** Drop-in `ImageSecretLoss` (YUV+LPIPS+FFL) replacing pure-L1 host loss in `exp0_inn_train.py`, with default weights and the perceptual quality term gated by the existing curriculum α-ramp; bit pressure preserved via `secret_weight=20.0`.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Motivation and approach

The baseline trains with `loss = img_weight * F.l1_loss(container, cover) + bit_weight * F.binary_cross_entropy_with_logits(logits, bits)`, ignoring `ImageSecretLoss` already shipped in `baseline/losses.py`. Per `idea.md`, baseline saturates `bit_acc` (≥0.999 on 12/13 attacks at 256), so the dominant lever for `score = 2*PSNR*(bit_acc-0.5)` is PSNR. This design swaps the L1 host term for the YUV+LPIPS+FFL perceptual bundle (default weights `yuv_w=1.5`, `lpips_w=1.0`, `ffl_w=1.5`), keeping `secret_weight=20.0` to retain the same BCE pressure that gave baseline `bit_acc_clean=1.000`. The perceptual quality term is multiplied by the curriculum's existing `attack_alpha`, so during the clean phase (`step < stage_a_steps`) the encoder is pure-bit-pressure (seeds the secret pathway just like baseline), and the perceptual quality pressure ramps in alongside the attack curriculum.

## 2. Files changed (Builder may only edit these)

- `code/baseline/exp0_inn_train.py` — wire `ImageSecretLoss` into the train loop, replacing the inline L1+BCE.
- `code/baseline/losses.py` — no algorithmic change; if needed, only add a tiny convenience helper. Default class definition is reused as-is.

Untouched (locked): everything in `infra/**`, `code/baseline/inn_model.py`.

## 3. CLI flag values for `baseline/exp0_inn_train.py` (config)

Locked (must NOT change): `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.

Tuned for this design:
- `--num_blocks 16` (baseline default)
- `--subnet_type rdb` (baseline default)
- `--lr 2e-4` (baseline default)
- `--img_weight 1.0` — REPURPOSED as the multiplier on `ImageSecretLoss`'s `l_quality` term (see §4); the Builder must wire it through accordingly.
- `--bit_weight 20.0` — REPURPOSED as `secret_weight` passed to `ImageSecretLoss`.
- `--stage_a_steps 5000` (baseline default; clean phase length)
- `--stage_b_steps 10000` (baseline default; α-ramp window)

All other flags keep their baseline defaults (`--ckpt_every 2000`, `--log_every 200`, `--val_every 1000`, `--val_n 50`, `--num_workers 4`, `--amp` on, `--seed 0`).

## 4. Algorithmic changes (concrete)

### 4.1 Instantiate the loss module once, after model construction

In `code/baseline/exp0_inn_train.py`, after the `model = INNCodec(...)` line and before the curriculum is built, add:

```python
from losses import ImageSecretLoss  # sibling, baseline-owned
img_loss_mod = ImageSecretLoss(
    secret_weight=args.bit_weight,   # 20.0
    yuv_w=1.5,
    lpips_w=1.0,
    ffl_w=1.5,
).to(device)
# LPIPS AlexNet is already frozen inside ImageSecretLoss.__init__.
```

### 4.2 Replace the inline loss computation in the training step

Inside the `with torch.amp.autocast(...)` block, the existing:
```python
bit_loss = F.binary_cross_entropy_with_logits(logits, bits)
img_loss = F.l1_loss(container, cover)
loss = args.img_weight * img_loss + args.bit_weight * bit_loss
```
must be replaced by:
```python
# alpha here is the *quality* alpha. We tie it to the existing curriculum
# attack_alpha so perceptual pressure ramps in alongside the attack
# curriculum: clean phase = pure bit pressure, full = full quality term.
quality_alpha = args.img_weight * attack_alpha
loss, loss_logs = img_loss_mod(
    cover=cover,
    stego=container,
    secret=bits,
    logits=logits,
    alpha=quality_alpha,
)
bit_loss = loss_logs["l_bce"]
img_loss = loss_logs["l_quality"]   # for backward-compatible logging
```

Notes:
- `img_loss_mod.forward` already does `secret_weight * BCE + alpha * L_quality`, so this single call replaces both terms; do NOT add `args.bit_weight * bit_loss` again or the BCE term will be double-counted.
- `quality_alpha = args.img_weight * attack_alpha` makes `--img_weight` the perceptual-loss scale knob (default 1.0). It is `0` while `step < stage_a_steps` (clean phase), exactly matching the curriculum's clean phase semantics.

### 4.3 LPIPS / FFL precision

The LPIPS module expects fp32 inputs and the AMP `autocast` context will downcast to fp16. To avoid LPIPS-AlexNet running in fp16 (numerically wobbly), wrap only the perceptual loss call in an `autocast(enabled=False)` block:
```python
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
```
This preserves AMP for the INN forward/backward but keeps the loss numerically stable.

### 4.4 Logging additions

Extend the per-`log_every` `rec` dict in `metrics.jsonl` to include the new loss components, sourced from `loss_logs`:
- `l_yuv` (float) — `loss_logs["l_yuv"].item()`
- `l_lpips` (float) — `loss_logs["l_lpips"].item()`
- `l_ffl` (float) — `loss_logs["l_ffl"].item()`
- `l_quality` (float) — `loss_logs["l_quality"].item()`
- `quality_alpha` (float)

Keep all existing fields (`bit_acc`, `psnr`, `l1_container_cover`, `secret_rms`, etc.) unchanged. The `l1_container_cover` diagnostic remains computed exactly as in baseline (independent of the loss change) so PSNR/L1 comparisons against baseline remain apples-to-apples.

## 5. Expected behavior after the change

- `metrics.jsonl` lines now include `l_yuv`, `l_lpips`, `l_ffl`, `l_quality`, `quality_alpha`.
- `val_metrics.jsonl` schema is unchanged (validation logic is untouched).
- `bit_acc_ema` should still climb to ≥0.999 by `step ~= 15000` (same `secret_weight=20.0` as baseline).
- `psnr_ema` is expected to be ≥ baseline's `psnr_ema` at convergence (target: +0.5 to +1.5 dB at step 30000).
- `bit_acc_clean = 1.000` at every eval resolution must hold post-eval.
- Wallclock: LPIPS-AlexNet at 256² adds ~5–10 ms/step; 30k steps remain well within a 12 h 2080Ti walltime.
- `score = 2 * psnr_mean * (bit_acc_mean - 0.5)` is expected to improve over baseline 27.27.

## 6. Constraints and invariants Builder must preserve

- Do NOT modify any `infra/**` file. (Locked.)
- Do NOT change locked CLI flag defaults (`--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`) and do NOT change the eval attack suite.
- Keep `bit_acc_clean = 1.000` achievable — `--bit_weight 20.0` (= `secret_weight`) must not be reduced.
- Preserve gradient clip `clip_grad_norm_(..., 5.0)`, optimizer (AdamW, betas (0.9, 0.999), weight_decay 0), cosine LR schedule, AMP setup, and the resume/checkpoint logic verbatim.
- Preserve the `container = container.clamp(-1.0, 1.0)` step before the attack and before passing to `ImageSecretLoss` (LPIPS/YUV/FFL all expect roughly [-1, 1]).
- Preserve curriculum semantics: `attack_alpha == 0` during the clean phase; therefore `quality_alpha == 0` during the clean phase too.
- All edits must live inside `code/baseline/`; no edits anywhere else.

## 7. Edge cases

- Resume from `ckpt_latest.pt`: `ImageSecretLoss` parameters are all frozen (LPIPS-AlexNet) or buffers; they are NOT in `model.state_dict()`. They are reinitialised on every process start, which is correct (no state to restore).
- AMP: only the loss call runs in fp32; the rest of the step keeps AMP. The unscale/clip/step/update sequence is unchanged.
- If `lpips` or `focal_frequency_loss` import fails, surface the error at startup (the `ImageSecretLoss(...).to(device)` call will fail fast); do not try/except around it.
