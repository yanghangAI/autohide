## design003
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS, check-scope PASS, verify-claims PASS — 9/9 snippets)
**Algorithm fidelity:** Combined design001 + design002 edits applied verbatim.
- `code/losses.py:47-67` `_gaussian_kernel_1d`/`_gaussian_blur_luma` (from d001).
- `code/losses.py:88-105` `_sobel_mag_luma` (from d002, differentiable, no `@torch.no_grad`).
- `code/losses.py:126-158` `multiscale_sobel_jnd_mask` (from d001, `@torch.no_grad`, per-scale 99-pct normalize then `amax` aggregate, `.detach()`).
- `code/losses.py:169-197` `__init__` accepts both `jnd_sigmas`/`jnd_pct` AND `tv_w`/`tv_eps`; assert allows `"sobel_multiscale"`.
- `code/losses.py:211-251` forward dispatches mask via `jnd_mode` (incl. `sobel_multiscale` branch at line 216-217), computes `w = 1/(1 + β·M_multi)`, then both masked-YUV, masked-L1, AND TV term — all reusing the *same* `w` (single mask compute, as required by design003 §6 critical-composition note). The TV's `w_norm` is therefore derived from the multi-scale mask, fulfilling §4.1 "Critical composition detail".
- `code/losses.py:260-264` `l_quality` includes both `pixel_w*l_pixel` and `tv_w*l_tv` additively.
- `code/losses.py:267-279` `logs` includes both `l_pixel` and `l_tv`.
- `code/exp0_inn_train.py:258-269` constructs `ImageSecretLoss` with `jnd_mode="sobel_multiscale"`, `jnd_sigmas=(1.0, 2.0, 4.0)`, `jnd_pct=0.99`, `tv_w=0.3`. All other kwargs match parent.
- `code/exp0_inn_train.py:458-462` logs all five new descriptor/metric fields. Smoke `metrics.jsonl` step-200 confirms `jnd_mode="sobel_multiscale"`, `jnd_sigmas=[1.0,2.0,4.0]`, `jnd_pct=0.99`, `l_tv≈0.128`, `tv_w=0.3`.
**Training-signal sanity:**
- Single mask compute reused across YUV / L1 / TV — no doubled helper cost; semantic-preserving composition.
- Multi-scale `w` shape `(B,1,H,W)`, value range identical regime as single-scale — TV `w_norm = w / w.amax` works without special-casing.
- TV gradient still flows through `g_res = _sobel_mag_luma(stego-cover)`; `w` (and therefore `w_norm`, `comp`) is detached via `@torch.no_grad` upstream, so the TV gradient targets only `stego` in the residual gradient (correct).
- All parent invariants preserved: `lpips_w=0.0` short-circuit, optimizer/LR/AMP/clip/curriculum unchanged (`exp0_inn_train.py` outside the loss-construction lines is the parent's file).
- Smoke run completed step 200 (~38s), `bit_acc≈0.94`, both `l_pixel` and `l_tv` produce sane values; the additional TV+multiscale cost did not destabilize training.
**Strongest objection:** This design carries the largest deviation surface from the parent (mask shape change AND new term), so failure-mode interactions are hardest to predict. However, both subcomponents are softer than (e.g.) deep perceptual terms, and design §5 explicitly bounds the worst case to "score < parent" not training collapse. The smoke run confirms no instability at step 200, and the design has the explicit retune-trigger guidance (reduce `tv_w` first if `bit_acc_clean<0.99` in real training). No code-level concern.
**Fixes required:** none.
