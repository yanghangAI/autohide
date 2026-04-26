## design003
**Verdict:** APPROVED

**Feasibility evidence:**
- `baseline/inn_model.py:103-166` (PatchBitAdapter) exposes the methods being replaced.
- `_make_sign_mask` at `baseline/inn_model.py:96` is the named placement anchor for `_make_hadamard_orthonormal`.
- `NUM_SLOTS = 1024` at `baseline/inn_model.py:41` is a power of two, satisfying the Sylvester precondition.
- The design preserves `repeat_interleave` upsampling (matches `baseline/inn_model.py:152`) and `adaptive_avg_pool2d(., (GRID, GRID))` decode (matches `baseline/inn_model.py:160`), so the tile↔mean-pool identity composes correctly.
- Step-0 anchor at `baseline/exp0_inn_train.py:249` is present.
- `(B, 1024) @ (1024, 1024)` matmul is straightforward; `.to(scrambled.dtype)` cast handles AMP.

**Idea contradiction check:** `idea.md` direction (3) prescribes "Walsh–Hadamard 1024-slot spreading … `(B, 1024) → matmul(H_1024, sign_masked_bits) → reshape to (32, 32) → upsample to DWT plane`" with the orthogonal-Hadamard inverse at decode. Design implements exactly this (with explicit `1/sqrt(1024)` orthonormal scaling, which idea.md mentions in invertibility terms). No contradiction.

**Strongest objection:** Persisting both `hadamard` and `hadamard_t` (≈8 MiB total) instead of reconstructing one from the other is mildly wasteful but avoids a `.t().contiguous()` per forward, which is a reasonable speed/memory tradeoff. The orthonormal scaling is the genuine correctness hinge: `H @ H^T = I` only after dividing by `sqrt(1024)`; if Builder substitutes the unscaled Sylvester form, decode logits scale by 1024 and the BCE operating point shifts, but sign recovery still works — so even a Builder mistake here would not break `bit_acc_clean = 1.000`. The §4.4 numeric (1e-3) assertion catches scaling slips immediately. I rule this out: the spec is unambiguous and the assertion is loud.

**Locked-scope check:** Files modified are `code/inn_model.py` and `code/exp0_inn_train.py`. No `infra/**` edits. No locked CLI flags changed.

**Parent consistency:** Parent `baseline/` is `Done`.
