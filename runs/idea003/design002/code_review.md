## design002
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS; check-scope PASS vs parent runs/idea001/design003; verify-claims PASS, 5 snippets)
**Algorithm fidelity:** Implementation matches design.md §4.1–§4.2.
- `code/losses.py:31-47` defines `variance_jnd_mask(cover, win=7)` under `@torch.no_grad()`: luma in [0,1] from `rgb_to_yuv`, uniform 7x7 kernel, mean and mean_sq via `F.conv2d` with `padding=3`, `var=(mean_sq-mean*mean).clamp_min(0.0)`, `std=sqrt(var+1e-12)`, per-image max-normalize with `clamp_min(1e-6)`, returns `.detach()`. The required `clamp_min(0.0)` guard is present.
- `code/losses.py:65-75` constructor adds `jnd_beta=0.0`, `jnd_mode="variance"`, with the design-mandated assert `jnd_mode in ("sobel","variance")`.
- `code/losses.py:91-100` weighted YUV: `sq=(yuv_stego-yuv_cover)**2; l_yuv_unweighted=sq.mean()`, then masked branch builds `M=variance_jnd_mask(cover, win=7)`, `w=1/(1+jnd_beta*M)`, `l_yuv=(w*sq).mean()`. The `else: # "sobel"` branch at line 95-96 also calls `variance_jnd_mask` rather than a real Sobel — design.md §4.1 explicitly permits this: "If only one helper is added, only the `variance` branch needs to compile/run." This design's training path only exercises `jnd_mode="variance"`, so the substitution is benign.
- `code/exp0_inn_train.py:258-263` (per grep) passes `jnd_beta=4.0, jnd_mode="variance"`.
- Per-step record adds `l_yuv_unweighted`, `jnd_beta=4.0`, `jnd_mode="variance"`.
**Training-signal sanity:**
- Mask detached under `torch.no_grad()`, computed from `cover` only; no autograd leakage.
- Variance-from-conv numerical recipe is the standard `E[X^2]-E[X]^2` form with `clamp_min(0.0)` before `sqrt`, preventing NaN on near-uniform regions.
- step 200 metrics: `l_yuv_unweighted=0.0826`, `l_yuv=0.0570` (~31% reduction) — within the predicted 30-60% range; bit_acc=0.936 at step 200 (mid-curriculum), bit_acc_ema=0.91. Consistent with `bit_acc_clean=1.000` being achievable at full 30k steps.
- LPIPS off; secret_weight=20.0; FFL untouched; optimizer/AMP/curriculum unchanged.
- `w` shape `(B,1,H,W)` broadcast across 3 YUV channels as intended.
- Edge case: constant-luma cover degenerates to `M~=1` everywhere (per design §7) due to `flat_max~=1e-6` clamp — design acknowledges this as harmless on COCO.
**Strongest objection:** The unused `else: # "sobel"` branch silently falls through to `variance_jnd_mask`. If a future caller mistakenly sets `jnd_mode="sobel"`, the assert passes but the actual mask is variance-based — a latent bug. However the design.md explicitly authorizes this shortcut for design002, the assert prevents arbitrary string values, and the only training-path caller in `exp0_inn_train.py` hard-codes `"variance"`. Within scope of this design, no live mis-dispatch can occur.
**Fixes required:** none.
