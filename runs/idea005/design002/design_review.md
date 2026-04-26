## design002
**Verdict:** APPROVED

**Feasibility evidence:**
- `baseline/inn_model.py:103-166` defines `PatchBitAdapter` with the exact `encode/decode` signatures the design preserves; `_make_sign_mask` at `baseline/inn_model.py:96-100` is the named anchor for inserting `_make_gaussian_kernel2d` before the class.
- The design's encode keeps the existing `repeat_interleave` tile (current implementation at `baseline/inn_model.py:152` is `tiled = grid.repeat_interleave(rep_h, dim=2).repeat_interleave(rep_w, dim=3)`) — exact match.
- `F.conv2d` with `groups=C` is autocast-safe and accepts `(B, 12, H, W)`-shaped input at any DWT spatial size; reflect-pad via `F.pad(..., mode="reflect")` is supported in PyTorch.
- Step-0 hook anchors after `print(f"[model] INNCodec n_params=...")` at `baseline/exp0_inn_train.py:249` — confirmed present.

**Idea contradiction check:** `idea.md` direction (2) describes "tile as today, then convolve the resulting 12-channel carrier with a fixed low-pass kernel … decode applies the matched kernel before the 32×32 mean-pool". Design implements precisely this with σ=2 DWT-px Gaussian, 9×9 kernel. No contradiction.

**Strongest objection:** The §4.3 invertibility argument is weaker than for orthogonal designs (001/003/004): convolving a piecewise-constant 4×4 block with a 9×9 Gaussian *does* perturb the per-block mean near boundaries, even with reflect padding — the matched filter's inverse is not the identity, it doubles the smoothing. The design admits this (≤1e-3 absolute error) and falls back on sign recovery for the hard `bit_acc_clean = 1.000` condition. Sign recovery is plausible because slot magnitude (~1) ≫ smoothing residual (~1e-3), but unlike the orthonormal designs this isn't *exactly* invertible — it's "exactly sign-correct on clean by margin." The step-0 sign assertion is therefore the actual safety net rather than an empty formality. I'm ruling this in as acceptable because (a) the margin is ~1000×, (b) the assertion catches violations loudly, and (c) the eval suite measures sign agreement, not numerical equality. Builder must keep the assertion.

**Locked-scope check:** Files modified are `code/inn_model.py` and `code/exp0_inn_train.py`. No `infra/**` edits. No locked CLI flags changed.

**Parent consistency:** Parent `baseline/` is `Done`.
