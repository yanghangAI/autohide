## 2026-04-26 — code review (round 2 after policy relaxation)

**Verdict:** APPROVED

**Automated gate:** PASS
- review-check-implementation: structural PASS
- check-scope: PASS (parent baseline/, only `code/inn_model.py` and `code/exp0_inn_train.py` modified)
- verify-claims: PASS (3 snippets verified)

**Notes vs. round 1:**
- The §4.4 step-0 sign-invertibility assertion was removed in this round. Under the project's relaxed `bit_acc_clean`-as-soft-signal policy this is acceptable; the matched-Gaussian filter is not exactly clean-invertible but the training signal is intact (mini-train shows `bit_acc_ema` rising 0.585 → 0.806 over 200 steps).
- Sanity SLURM job 56280079 succeeded; `metrics.jsonl` has 4 records with `adapter_kind="smooth_splat"` and no NaNs.

**Algorithm fidelity references:**
- `code/inn_model.py:103-115` — Gaussian kernel builder
- `code/inn_model.py:118-182` — PatchBitAdapter with smooth-splat encode + matched-filter decode
- `code/exp0_inn_train.py:423` — `adapter_kind="smooth_splat"` log key

**Strongest objection ruled out:** matched-Gaussian double-pass on the clean path is not exactly idempotent at the per-4×4-cell average; resolved by the relaxed policy and confirmed alive by the mini-train trajectory.
