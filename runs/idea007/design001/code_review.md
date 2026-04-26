## design001
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS, check-scope PASS, verify-claims PASS — 6/6 snippets)
**Algorithm fidelity:** Implementation matches design.md §4.1/§4.2.
- `code/losses.py:47-67` adds `_gaussian_kernel_1d` and `_gaussian_blur_luma` (separable, replicate-padded) per design §4.1.
- `code/losses.py:111-143` defines `multiscale_sobel_jnd_mask` with `@torch.no_grad`, per-scale 99th-percentile normalization (`q = torch.quantile(flat.float(), pct, dim=1).clamp_min(1e-6)`) applied BEFORE per-pixel max-aggregation (`torch.stack(...).amax(dim=0)`). Order matches design constraint §6 ("normalize each scale's magnitude *before* the per-pixel max-aggregation").
- `code/losses.py:154-178` extends `__init__` accepting `jnd_mode in ("sobel","variance","sobel_multiscale")` with `jnd_sigmas`/`jnd_pct` kwargs.
- `code/losses.py:192-214` dispatches on `jnd_mode` and routes to `multiscale_sobel_jnd_mask(cover, sigmas=self.jnd_sigmas, pct=self.jnd_pct)`. The same `w` is used for both YUV term and masked-L1 — single mask compute per forward, as required.
- `code/exp0_inn_train.py:258-268` constructs `ImageSecretLoss` with `jnd_mode="sobel_multiscale"`, `jnd_sigmas=(1.0, 2.0, 4.0)`, `jnd_pct=0.99`; all other kwargs (`secret_weight=20.0`, `yuv_w=1.5`, `lpips_w=0.0`, `ffl_w=1.5`, `jnd_beta=4.0`, `pixel_w=0.5`) match parent verbatim.
- `code/exp0_inn_train.py:457-459` logs `jnd_mode="sobel_multiscale"`, `jnd_sigmas=[1.0,2.0,4.0]` (list, not tuple — JSON-safe), `jnd_pct=0.99`. Smoke `metrics.jsonl` step-200 record confirms all three fields plus `l_yuv`, `l_yuv_unweighted`, `l_pixel`, `jnd_beta=4.0`, `pixel_w=0.5`.
**Training-signal sanity:**
- Mask is `@torch.no_grad` and final `.detach()` — weight only, no gradient source. Cover-only input as required.
- `w = 1.0/(1.0 + jnd_beta * M)` always in (0,1] since M ≥ 0; broadcasts (B,1,H,W) over (B,3,H,W) sq cleanly.
- Existing `sobel_jnd_mask`/`variance_jnd_mask` retained unchanged.
- LPIPS short-circuit (`if self.lpips_w == 0.0`) preserved.
- FP32 cast for `torch.quantile` matches design edge-case §7 (AMP safety).
- Smoke run reached step 200 (38s wallclock), `bit_acc≈0.93` rising and `l_yuv` numerically reasonable (~0.049, similar order to expected).
**Strongest objection:** The 99th-percentile normalization permits ~1% of mask pixels to exceed 1, so `M_multi.amax` can exceed 1 — slightly different regime than the parent's exact `[0,1]`. However, `w=1/(1+βM)` remains in (0,1] for any M≥0, and the design explicitly anticipates and accepts this in §7. No correctness issue.
**Fixes required:** none.
