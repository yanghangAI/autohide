## 2026-04-26 — Design review: APPROVED

Reviewer: design review pass. Feasibility verified against `baseline/inn_model.py:103-166` (PatchBitAdapter shape) and `baseline/exp0_inn_train.py:249` (post-construction logging anchor). Strongest objection: spec helper has Python-syntax-invalid arg ordering (non-default after default); call sites use kwargs so the reorder is mechanical. Parent `baseline/` is Done.
