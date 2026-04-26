## 2026-04-26 — Design review
**Verdict:** APPROVED
Reviewed single-scale Sobel mask + edge-aware TV penalty on residual luma (tv_w=0.3). Parent runs/idea003/design003 is Done. All prescribed edits feasible. TV term operates on residual (not stego) and is differentiable; mask is reused (one mask compute per forward). No infra/locked-flag violations. See `design_review.md` for evidence.
