## design001
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS; check-scope PASS vs parent runs/idea001/design003; verify-claims PASS, 5 snippets)
**Algorithm fidelity:** Implementation matches design.md §4.1–§4.2 exactly.
- `code/losses.py:48-67` defines `sobel_jnd_mask` under `@torch.no_grad()`: luma extracted via `rgb_to_yuv(cover)[:,0:1]` rescaled to [0,1], Sobel-X/Y conv with `padding=1`, `sqrt(gx*gx+gy*gy+1e-12)`, 5x5 Gaussian blur (`padding=2`), per-image max-normalize with `clamp_min(1e-6)`, returns `.detach()`. Identical to the design spec.
- `code/losses.py:102-111` replaces unweighted YUV MSE with `sq=(yuv_stego-yuv_cover)**2`, `l_yuv_unweighted=sq.mean()`, then `M=sobel_jnd_mask(cover); w=1/(1+jnd_beta*M); l_yuv=(w*sq).mean()` when `jnd_beta>0`, else falls back. `w` shape `(B,1,H,W)` broadcasts over the 3 YUV channels as designed.
- `code/losses.py:127-128` adds `l_yuv_unweighted` to logs.
- `code/exp0_inn_train.py:258-263` calls `ImageSecretLoss(secret_weight=args.bit_weight, yuv_w=1.5, lpips_w=0.0, ffl_w=1.5, jnd_beta=4.0)`.
- `code/exp0_inn_train.py:448-450` records `l_yuv`, `l_yuv_unweighted`, `jnd_beta=4.0` in metrics.
**Training-signal sanity:**
- Mask is detached and computed under `torch.no_grad()` from `cover` only; no autograd path through the weight, encoder cannot game it.
- `l_yuv_unweighted` (0.157 -> 0.0894 -> 0.0799) > `l_yuv` (0.107 -> 0.064 -> 0.055) at steps 100/150/200, confirming the weight `w<=1` is biting and the ratio is in the predicted 30-60% reduction range.
- LPIPS short-circuit preserved (`l_lpips=0.0`); secret_weight=20.0 preserved; bit_acc reaches ~0.999 by step 150 of the 200-step smoke train, consistent with `bit_acc_clean=1.000` being attainable.
- AMP scope unchanged; helper kernels `.to(luma.device, luma.dtype)` so they match the fp32-wrapped autocast=False block. No NaN risk: `sqrt(gx*gx+gy*gy+1e-12)` and `clamp_min(1e-6)` on flat_max guard degenerate inputs.
- mask broadcasting `(B,1,H,W) * (B,3,H,W)` is intentional and applies same per-pixel weight to Y, U, V.
- test_output/metrics.jsonl is well-formed for steps up to 200; checkpoint emission path (the existing design003 path) is untouched.
**Strongest objection:** The luma channel used inside the Sobel mask is computed via `rgb_to_yuv(cover)`, where `rgb_to_yuv` operates inside `torch.no_grad()` so it cannot leak gradients. I confirmed `rgb_to_yuv` is a pure tensor op (no learnable params) at `code/losses.py:25` and that the call site is wrapped by `@torch.no_grad()` plus `.detach()` on return. Even if AMP were enabled, the helper runs in fp32 because it is invoked from within the `cover_fp32` autocast=False region inherited from design003. No leakage path identified.
**Fixes required:** none.
