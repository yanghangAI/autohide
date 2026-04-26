
---
# Review on 2026-04-26

## design001
**Verdict:** APPROVED

**Feasibility evidence:**
- `baseline/exp0_inn_train.py:246` constructs `model = INNCodec(...)`, the precise insertion point named in §4.1 — instantiating `ImageSecretLoss` here is mechanically possible.
- `baseline/exp0_inn_train.py:366-377` is the autocast block with `attack_alpha`, `bit_loss = F.binary_cross_entropy_with_logits(logits, bits)`, `img_loss = F.l1_loss(container, cover)`, `loss = args.img_weight * img_loss + args.bit_weight * bit_loss` — exactly the lines the design replaces in §4.2; `attack_alpha` is in scope (line 366) so `quality_alpha = args.img_weight * attack_alpha` is well-defined.
- `baseline/exp0_inn_train.py:371` already does `container = container.clamp(-1.0, 1.0)` before the loss, satisfying the `[-1,1]` precondition of LPIPS/YUV/FFL (§6 invariant).
- `baseline/losses.py:30-78` defines `ImageSecretLoss(secret_weight, yuv_w, lpips_w, ffl_w)` with `forward(cover, stego, secret, logits, alpha) -> (total, logs)` returning `l_bce`, `l_yuv`, `l_lpips`, `l_ffl`, `l_quality` — signatures and log keys called by the design exist verbatim.
- `baseline/exp0_inn_train.py:395` already nests `torch.amp.autocast("cuda", enabled=False)` for the diagnostic L1, confirming the fp32-wrap pattern (§4.3) is already idiomatic in the file.
- CLI flags `--img_weight` (line 211), `--bit_weight` (line 213), `--stage_a_steps` (line 217), `--stage_b_steps` (line 219) all exist in argparser.

**Idea contradiction check:** `idea.md` direction (1) prescribes drop-in `ImageSecretLoss` with default weights `yuv_w=1.5, lpips_w=1.0, ffl_w=1.5, secret_weight=20.0`, α tied to attack curriculum, clean phase = pure-bit pressure. Design001 sets exactly those weights, ties `quality_alpha = args.img_weight * attack_alpha`, and notes `quality_alpha == 0` during clean phase — direct elaboration, no contradiction.

**Strongest objection (ruled out):** Repurposing `--img_weight` and `--bit_weight` as `l_quality` multiplier and `secret_weight` (without renaming flags) could mislead a reader of `metrics.csv` flag dumps. Ruled out because (a) the design explicitly documents the repurpose in §3, (b) the numerical values (`1.0` and `20.0`) are unchanged from the baseline meaning of those flags so cross-run flag comparisons remain coherent, and (c) `secret_weight=20.0` keeps the BCE pressure that delivered baseline `bit_acc_clean=1.000`, so the hard condition is preserved.

**Fixes required:** N/A (APPROVED).
