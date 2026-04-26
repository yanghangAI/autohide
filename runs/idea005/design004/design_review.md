## design004
**Verdict:** APPROVED

**Feasibility evidence:**
- `baseline/inn_model.py:103-166` (PatchBitAdapter) and `baseline/inn_model.py:96` (`_make_sign_mask`) anchor the placements specified in §4.1-4.3.
- Constants at `baseline/inn_model.py:40-43` (`GRID=32`, `NUM_SLOTS=1024`, `INN_CHANNELS=12`) match the spec's assumptions.
- The tile/mean-pool round-trip uses the existing `repeat_interleave` (`baseline/inn_model.py:152`) and `adaptive_avg_pool2d(., (GRID,GRID))` (`baseline/inn_model.py:160`) — same identity composition as designs 001/003.
- 32×32 batched matmul `D @ S @ D.T` and its inverse `D.T @ X @ D` are mechanically feasible on `(B, 32, 32)` tensors. Scattered indexing `X[:, ky, kx] = scrambled` and `X[:, ky, kx]` gather work on a `(B, 32, 32)` tensor with `(n_bits,)` index tensors.
- Step-0 anchor at `baseline/exp0_inn_train.py:249` is present.

**Idea contradiction check:** `idea.md` direction (4) prescribes "Each bit ↔ one 2D-DCT basis function on a 32×32 frequency grid, restricted to the mid-band (exclude DC and the highest spatial frequencies). … upsampled to DWT plane. JPEG-friendly … Decode: 32×32 DCT of the pooled recovery, then read off the 896 selected mid-band coefficients." Design implements exactly this with deterministic mid-band annulus selection. No contradiction.

**Strongest objection:** The mid-band annulus calculation in §4.2 has a self-consistency wrinkle: the spec text says "For grid=32, r_lo=0.20, r_hi=0.85 the annulus contains roughly 696" then immediately recommends `r_lo=0.10, r_hi=0.90` (yielding > 950) as the "designer-recommended default." The class-level constants `DCT_R_LO=0.10, DCT_R_HI=0.90` in §4.3 commit to the recommended values, and `_select_midband_coeffs` raises if it can't supply 896 — so the failure mode is loud and at construction time, not silent. I checked the math: with grid=32, the radius normalization `r = sqrt(k_y^2 + k_x^2) / sqrt(2*(grid-1)^2)` puts max r = 1 at (31,31). At r_lo=0.10, only the (0,0)–(0,4)/(4,0) corner is excluded (≈ a few coefficients), so well over 950 remain. The construction is feasible. Secondary concern: the §4.5 numeric error formula `(_rec - (_bits*2-1) * sign_mask * sign_mask).abs().max()` — `sign_mask*sign_mask = 1` so it reduces to `(_rec - (_bits*2-1)).abs().max()`, which matches the orthonormal-DCT prediction. Correct, just clumsy.

**Locked-scope check:** Files modified are `code/inn_model.py` and `code/exp0_inn_train.py`. No `infra/**` edits. No locked CLI flags changed.

**Parent consistency:** Parent `baseline/` is `Done`.
