## 2026-04-26 — Design review: APPROVED

Reviewer: feasibility verified at `baseline/inn_model.py:103-166` and `baseline/exp0_inn_train.py:249`. Strongest objection: matched Gaussian filter is not exactly invertible (only sign-perfect by ~1000× margin); design's step-0 sign assertion is a real safety net, must be preserved. Parent `baseline/` Done.
