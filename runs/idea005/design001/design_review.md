## design001
**Verdict:** APPROVED

**Feasibility evidence:**
- `baseline/inn_model.py:103-166` exposes the full `PatchBitAdapter` class with `__init__(self, n_bits=896)`, `encode(bits, h_dwt, w_dwt)`, and `decode(tensor_rec)` — exactly the names the design preserves. Line 130 (`sign = _make_sign_mask(NUM_SLOTS)`) and the `register_buffer("sign_mask", sign, persistent=True)` on line 131 are reused unchanged.
- `baseline/inn_model.py:96-100` defines `_make_sign_mask(...)`; the design adds `_make_balanced_permutation` immediately after it (anchor exists).
- Constants `GRID = 32`, `NUM_SLOTS = 1024`, `INN_CHANNELS = 12` are defined at `baseline/inn_model.py:40-43` and are imported into the spec's reasoning.
- The step-0 invertibility hook anchors after `print(f"[model] INNCodec n_params=...")` at `baseline/exp0_inn_train.py:249` — line exists.
- `index_select` over a flat (h*w) layout and a 1D `(NUM_SLOTS*p,)` permutation is mechanically possible at the named tensor shapes. The encode produces `(B, 1, h_dwt, w_dwt)` then `.expand(B, INN_CHANNELS, h_dwt, w_dwt)`, matching the contract that `extract/embed` see `(B, 12, H/2, W/2)`.

**Idea contradiction check:** `idea.md` direction (1) explicitly names a "fixed deterministic random permutation of the 16,384 DWT-spatial positions across the 1024 slots, balanced so each slot owns exactly 16 positions" — the design implements exactly this. No contradiction.

**Strongest objection:** The Python helper signature `def _make_balanced_permutation(num_slots: int = NUM_SLOTS, num_positions: int, ...)` places a non-default keyword argument after a defaulted one, which is a `SyntaxError`. Builder will trivially reorder (`num_positions: int` first, or make all args keyword-only via `*`); this is a code-typo in spec text, not a design defect. The algorithm itself is sound, and call sites in §4.2 use keyword arguments (`num_slots=NUM_SLOTS, num_positions=..., seed=...`) so reordering does not propagate. I rule this out as a non-blocking spec typo. Secondary concern: persisting only the canonical 128×128 permutation while rebuilding non-canonical resolutions at runtime makes the buffer set resolution-dependent at eval time — but the rebuild seed is deterministic (`0xD1FF5E ^ (h*1000003 + w)`), so eval is reproducible across processes. Acceptable.

**Locked-scope check:** Files modified are `code/inn_model.py` and `code/exp0_inn_train.py` (both inside the design's own `code/` mirror, not under `infra/**`). No locked CLI flags changed. Eval suite untouched.

**Parent consistency:** Parent `baseline/` is `Done` (step 30000, score 27.27 in `results.csv`).
