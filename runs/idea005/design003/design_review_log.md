## 2026-04-26 — Design review: APPROVED

Reviewer: feasibility verified at `baseline/inn_model.py:103-166`, `:41` (NUM_SLOTS=1024 power of 2), `:152/160` (tile/pool identity), and `baseline/exp0_inn_train.py:249`. Strongest objection: 8 MiB pre-transposed buffer is mildly wasteful but a reasonable speed tradeoff; orthonormal scaling is the correctness hinge and the 1e-3 numeric gate guards it. Parent `baseline/` Done.
