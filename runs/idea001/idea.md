**Idea Name:** Perceptual image-loss rebalancing for headroom-driven PSNR gains

**Approach:** Replace the baseline's pure-L1 host-reconstruction loss with the already-implemented `ImageSecretLoss` (YUV + LPIPS + FFL) in `baseline/losses.py`, exploiting the baseline's saturated bit_acc (~0.999) to spend that headroom on higher PSNR / perceptual quality and lift `score = 2*PSNR*(bit_acc-0.5)`.

**Expected Designs:** 3

**Suggested Parent:** baseline/

**Baseline Source:** baseline/ (only `Done` entry in `results.csv`; score=27.27, PSNR=27.33, bit_acc=0.999, bit_acc_clean=1.0)

**Relationship to prior work:** New axis — only the baseline exists in `results.csv`. The baseline trains with `F.l1_loss(container, cover)` plus BCE bit loss and ignores the rich `ImageSecretLoss` (YUV/LPIPS/FFL) module already shipped in `baseline/losses.py`. Baseline saturates bit_acc on 12 of 13 attacks at every resolution (only `chain_wechat` dips to ~0.988 at 256), so the dominant lever for `score` is PSNR rather than bit_acc, and a perceptual-loss formulation that distributes residuals into perceptually-tolerated frequency/chroma bands is the natural first axis to test before architectural changes.

**Motivation from the data:**
- Baseline 256×256: PSNR=26.75, bit_acc_min=0.989 (chain_wechat), all others ≥0.999.
- `score = 2 * PSNR * (bit_acc - 0.5)` is roughly linear in PSNR at this regime, so a +1 dB PSNR gain is worth ~+1.0 score, while squeezing the last 0.001 of bit_acc is worth ~+0.05.
- L1 minimizes pixel error uniformly; YUV+LPIPS+FFL preferentially preserves luma/chroma and perceptual structure, leaving the encoder free to push residual energy into high-frequency/chroma bands that survive the JPEG/resize attack mix and are also perceptually invisible (= higher PSNR for the same bit_acc, or same PSNR with less perceptually-visible artifacts).

**Suggested design directions (Designer to refine):**
1. Drop-in `ImageSecretLoss` with default weights (yuv_w=1.5, lpips_w=1.0, ffl_w=1.5, secret_weight=20.0), `alpha(t)` tied to the existing α-ramp curriculum so the perceptual term ramps in alongside the attack curriculum (clean phase = pure-bit pressure to seed the secret pathway).
2. LPIPS-dominant variant — emphasize lpips_w (e.g. 3.0) and shrink yuv_w/ffl_w, testing whether perceptual-only quality (which doesn't reward PSNR directly) still raises measured PSNR via residual reshaping.
3. YUV+FFL only (no LPIPS) — cheaper, no extra forward pass through AlexNet, sanity-checks whether the wins come from chroma/frequency reweighting alone vs. requiring the deep-feature term.

**Constraints:**
- Parent is `baseline/` (only `Done` entry available); designs may modify `baseline/exp0_inn_train.py` and `baseline/losses.py`, must not change locked hyperparameters (`--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`) or any `infra/` file.
- LPIPS adds a frozen AlexNet forward per step; Designer should confirm 30k steps still fit in ~12 h on a 2080Ti (baseline run finished comfortably within walltime; LPIPS-AlexNet at 256² adds ~5–10 ms/step).
- Hard condition `bit_acc_clean = 1.000` at every eval resolution must hold — Designer should keep `secret_weight` ≥ a value that preserves bit_acc_clean=1.0 (baseline used bit_weight=20.0; matching `secret_weight=20.0` is a safe starting point).
