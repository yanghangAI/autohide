## design002
**Verdict:** APPROVED

**Feasibility evidence:**
- `baseline/exp0_inn_train.py:246` (`model = INNCodec(...)`) is the documented insertion point for `ImageSecretLoss` — constructor signature `ImageSecretLoss(secret_weight=…, yuv_w=…, lpips_w=…, ffl_w=…)` matches `baseline/losses.py:38-47`, so passing `yuv_w=0.5, lpips_w=3.0, ffl_w=0.5, secret_weight=args.bit_weight` is feasible.
- `baseline/exp0_inn_train.py:366-377` is the L1+BCE block to be replaced; `attack_alpha` (line 366) and `args.img_weight` (line 211) are both in scope for `quality_alpha = args.img_weight * attack_alpha`.
- `baseline/losses.py:53-67` computes `total = self.secret_weight * l_bce + alpha * l_quality` with `l_quality = yuv_w*l_yuv + lpips_w*l_lpips + ffl_w*l_ffl`, so the `(3.0, 0.5, 0.5)` reweighting is mechanical — no architectural change required.
- `container = container.clamp(-1.0, 1.0)` at `baseline/exp0_inn_train.py:371` satisfies the LPIPS `[-1,1]` precondition.
- Existing nested `autocast("cuda", enabled=False)` at line 395 confirms the fp32-wrap pattern is supported.

**Idea contradiction check:** `idea.md` direction (2) explicitly proposes "LPIPS-dominant variant — emphasize lpips_w (e.g. 3.0) and shrink yuv_w/ffl_w." Design002 implements precisely `lpips_w=3.0, yuv_w=0.5, ffl_w=0.5` — direct, faithful elaboration.

**Strongest objection (ruled out):** A higher LPIPS weight inside the same `quality_alpha` envelope could in principle distort gradients away from BCE enough to threaten `bit_acc_clean=1.000`. Ruled out because `secret_weight=20.0` is unchanged from baseline (which achieved `bit_acc_clean=1.000`), and the perceptual term is gated to `0` during the entire 5000-step clean phase via `quality_alpha = args.img_weight * attack_alpha`, giving the BCE pathway a long head-start.

**Fixes required:** N/A (APPROVED).
