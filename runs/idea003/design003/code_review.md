## design003
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS; check-scope PASS vs parent runs/idea001/design003; verify-claims PASS, 6 snippets)
**Algorithm fidelity:** Implementation matches design.md §4.1–§4.2.
- `code/losses.py:48-...` defines `sobel_jnd_mask` (same recipe as idea003/design001: `@torch.no_grad`, Sobel X/Y, 5x5 Gaussian, per-image max-normalize, `.detach()`).
- `code/losses.py:71-...` also defines `variance_jnd_mask` for symmetry (acknowledged in the file comment "design002; this design uses jnd_mode='sobel'").
- `code/losses.py:99-112` constructor adds `jnd_beta=0.0`, `pixel_w=0.0`, `jnd_mode="sobel"` with the `("sobel","variance")` assert.
- `code/losses.py:130-148` masked branch: `M=sobel_jnd_mask(cover)` when mode=="sobel", `w=1/(1+jnd_beta*M)`, `l_yuv=(w*sq).mean()`, then **the same `w` is reused** for `l_pixel=(w*(stego-cover).abs()).mean()` — exactly as the design mandates ("MUST be computed exactly once per `forward` call and reused"). The L1 is taken in RGB space (`stego - cover`), not YUV, matching the explicit design constraint.
- `code/losses.py:157-160` `l_quality = yuv_w*l_yuv + lpips_w*l_lpips + ffl_w*l_ffl + pixel_w*l_pixel`.
- `code/losses.py:166-171` logs include `l_yuv`, `l_yuv_unweighted`, `l_pixel`, `l_quality`.
- `code/exp0_inn_train.py:258-265` constructs `ImageSecretLoss(..., jnd_beta=4.0, pixel_w=0.5, jnd_mode="sobel")`.
- `code/exp0_inn_train.py:450-455` records `l_yuv`, `l_yuv_unweighted`, `l_pixel`, `jnd_beta=4.0`, `pixel_w=0.5`, `jnd_mode="sobel"`.
**Training-signal sanity:**
- step 200 metrics: `l_yuv=0.0448`, `l_yuv_unweighted=0.0650` (~31% reduction, expected); `l_pixel=0.204` -> contribution `pixel_w*l_pixel = 0.102` to `l_quality=0.220`. The masked L1 is therefore *competitive* with the YUV term (`yuv_w*l_yuv = 1.5*0.0448 = 0.067`) but not dominant by >5x — within the design's "competitive, not dominant" target stated in §7.
- bit_acc=0.941 at step 200 (mid-curriculum, `alpha=1.0` in stage transition); ema 0.908. The added L1 has not collapsed the bit pathway in 200 steps; full 30k convergence to `bit_acc_clean=1.000` is plausible.
- Mask reused — only one `sobel_jnd_mask` call per forward; no doubled cost.
- Mask detached, computed under `torch.no_grad()`, from `cover` only. RGB-pixel L1 broadcasts the `(B,1,H,W)` mask over 3 channels intentionally.
- LPIPS short-circuit preserved (`l_lpips=0.0`); secret_weight=20.0, FFL term unchanged, optimizer/scheduler/AMP/curriculum untouched.
- The `pixel_w=0.0` and `jnd_beta=0.0` fallback paths are preserved (lines 144-148): defensive, not exercised in this design.
**Strongest objection:** The masked L1 contribution (~0.10) is roughly 1.5x larger than the masked YUV contribution (~0.067) at step 200. The design predicted the L1 would be "competitive, not dominant" with a >5x ratio as the threshold to flag concern. 1.5x is comfortably within the safe range, and it is the *masked* L1 — flat regions still get full weight, so the encoder still gets a signal to suppress flat-region residuals. PSNR at step 200 is 14.34 (design001 was 13.42), suggesting the extra pixel-domain signal is helping early. Risk that the L1 dominates later as YUV-MSE shrinks faster than L1: would only manifest after several thousand steps; flagging here would be premature given the design's 5x threshold.
**Fixes required:** none.
