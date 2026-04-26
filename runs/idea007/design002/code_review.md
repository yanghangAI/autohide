## design002
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS, check-scope PASS, verify-claims PASS — 6/6; one unattributed snippet skipped, no failure)
**Algorithm fidelity:** Implementation matches design.md §4.1/§4.2.
- `code/losses.py:70-82` adds `_sobel_mag_luma(rgb)` returning differentiable Sobel magnitude on luma — NOT decorated with `@torch.no_grad`, NO `.detach()`. Required so the TV term gradients flow back to `stego` (design §6 invariant).
- `code/losses.py:111-135` extends `__init__` with `tv_w=0.0`, `tv_eps=1e-6`. `jnd_mode` assert preserved as `("sobel","variance")` — design002 does NOT add `"sobel_multiscale"` (correct: this design is single-scale only).
- `code/losses.py:149-185` inside the `jnd_beta>0` branch: after computing `w` and `l_pixel`, a second `if self.tv_w > 0.0:` block computes `w_max = w.view(B,-1).amax(dim=1).clamp_min(self.tv_eps)`, then `w_norm = w/w_max[...]`, `comp = 1.0 - w_norm`, `g_res = _sobel_mag_luma(stego - cover)`, and `l_tv = (comp * g_res).mean()`. The same `w` (from a single mask compute) is reused — matches design §6 "mask computed exactly once and reused".
- `code/losses.py:194-198` adds `+ self.tv_w * l_tv` to `l_quality`. `code/losses.py:209` adds `l_tv=l_tv.detach()` to logs.
- TV operates on residual `(stego - cover)`, NOT `stego` directly — design §6 critical invariant satisfied.
- `code/exp0_inn_train.py:258-268` keeps `jnd_mode="sobel"` (single-scale) and adds `tv_w=0.3`. All other kwargs match parent.
- `code/exp0_inn_train.py:456-458` logs `jnd_mode="sobel"`, `l_tv=float(loss_logs["l_tv"])`, `tv_w=0.3`. Smoke `metrics.jsonl` step-200 confirms all fields, including `l_tv≈0.131`.
**Training-signal sanity:**
- TV gradient pathway: `_sobel_mag_luma` is differentiable; `comp` is detached (derived from no-grad `w`), but `g_res` is on the autograd graph. `mean(comp * g_res)` differentiates wrt `stego` correctly. The `+1e-12` inside `sqrt` prevents NaN gradient at flat-residual pixels.
- `comp = 1 - w_norm ∈ [0, ~0.8]` at β=4 as design §7 anticipates — penalty concentrates in flat regions, vanishes at maximum-textured pixel. Correct semantic.
- The else-branch (jnd_beta=0) fallback `l_tv = g_res.mean()` is unreachable in this configuration but preserved for ablation symmetry, matching design §7.
- `l_pixel` (magnitude penalty) and `l_tv` (gradient penalty) are non-redundant per design §7 — independent gradient pathways.
- Smoke run hit step 200, bit_acc≈0.93, `l_tv≈0.131` (somewhat larger than the design's predicted ~5e-3 estimate, since at step 200 the model has not converged and residuals are large; expected to shrink during training, no immediate concern).
**Strongest objection:** `l_tv` magnitude at step 200 (~0.131) is ~25× larger than the design's order-of-magnitude estimate (~5e-3). With `tv_w=0.3` this contributes ~0.04 to `l_quality` — ~4× larger than predicted. However, `l_quality` total (~0.265) is still dominated by other terms and the magnitude is consistent with un-converged residuals; the design's estimate assumes a converged ~1e-2 residual scale. No bug; the design itself permits this and flags `bit_acc_clean<0.99` as the only retune trigger.
**Fixes required:** none.
