## 2026-04-26 — Design review: APPROVED

Reviewer: feasibility verified at `baseline/inn_model.py:103-166`, `:40-43` (constants), `:152/160` (tile/pool), and `baseline/exp0_inn_train.py:249`. Strongest objection: spec mentions two annulus choices but commits to (0.10, 0.90) as class constants; the `_select_midband_coeffs` count check raises loudly at construction if <896 pairs available. Numeric formula is verbose but correct (`sign_mask*sign_mask=1`). Parent `baseline/` Done.
