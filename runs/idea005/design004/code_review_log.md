# Code Review Log — design004

---
## 2026-04-26 — Code review pass
## design004
**Verdict:** APPROVED

**Automated gate:** PASS (`review-check-implementation`: structural OK; `check-scope` PASS against parent `baseline/`; `verify-claims` 5/5 snippets matched)

**Algorithm fidelity:**
- `_make_dct_ii_orthonormal` at `code/inn_model.py:103-113` constructs the orthonormal DCT-II matrix with `D[0,m] = sqrt(1/n)`, `D[k,m] = sqrt(2/n) cos(pi(2m+1)k/(2n))`, exactly matching design §4.1. (Local `import math` inside the function is unusual but not incorrect.)
- `_select_midband_coeffs` at `code/inn_model.py:116-137` deterministically enumerates (k_y, k_x) within the radial annulus, sorts by ascending r then lex order, and raises if fewer than n_bits candidates — faithful to §4.2.
- `PatchBitAdapter.__init__` at `code/inn_model.py:158-174` uses class constants `DCT_R_LO=0.10`, `DCT_R_HI=0.90`, registers length-`n_bits` `sign_mask`, `dct_d`, `dct_dt`, and `coeff_idx` — matching §4.3 and §7.
- `_idct2d` and `_dct2d` at `code/inn_model.py:176-182` implement `D^T X D` (inverse) and `D S D^T` (forward) — correct 2D separable DCT.
- `encode` at `code/inn_model.py:184-202`: bipolar → sign-mask scramble (length n_bits) → scatter into (B, 32, 32) frequency grid at `coeff_idx` → 2D iDCT → tile to DWT → broadcast to 12 channels. Faithful to §4.3.
- `decode` at `code/inn_model.py:204-213`: pool → channel-mean → 2D DCT → read off `coeff_idx` → unscramble.
- Step-0 sign + 1e-3 numeric assertion at `code/exp0_inn_train.py:251-268` matches §4.5. Runtime log: `[adapter] dct-midband invertibility check: PASS (max err 8.34e-06)`.
- Logging keys `adapter_kind="dct_midband"`, `adapter_midband_lo=0.10`, `adapter_midband_hi=0.90` at `code/exp0_inn_train.py:441-443`, matching §5.

**Training-signal sanity:**
- Orthonormal DCT-II means `D D^T = I`; the round-trip is sign-perfect with max err ~8e-6 in fp32. `bit_acc_clean = 1.000` structurally guaranteed.
- Per-pixel spatial RMS after iDCT of 896 ±1 frequency coefficients is ~sqrt(896/1024)=0.935 — same scale as baseline's ±1 carrier, no loss-weight retuning needed.
- Mini-train ran to completion: PSNR 12.88 → 17.93 (best of the three idea005 designs at step 200), bit_acc 0.94–1.00, no NaN — healthy curve.
- Loss/optimizer/attack composition unchanged from baseline.

**Strongest objection:** The encode-side `_idct2d` returns `D^T @ X @ D` rather than the textbook 2D inverse of a 2D forward DCT formulated as `forward = D S D^T`, `inverse = D^T S D`. With the orthonormal D used here, `D D^T = I`, so the inverse of `forward(S) = D S D^T` is indeed `D^T (D S D^T) D = S` — i.e. the chosen `_idct2d`/`_dct2d` pair are mutual inverses. The empirical max err of 8.34e-06 confirms this. (I considered whether the ordering of D and D^T on the rows-vs-columns axes might silently transpose the output frequency grid, but since `coeff_idx` is used identically in encode-scatter and decode-read, any consistent transpose cancels.) Not a blocker.

**Fixes required:** none.
