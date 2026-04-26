## design003
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS, check-scope PASS vs parent runs/idea003/design003, verify-claims PASS on 9 snippets)
**Algorithm fidelity:** d003's design.md §4 mandates the d001 EMA edits verbatim plus a one-line constructor change (`pixel_w` 0.5 → 0.75) and an updated logged literal. Verified via `diff` against d001's `code/exp0_inn_train.py`: only two lines differ — line 320 (`pixel_w=0.75` in `ImageSecretLoss(...)`) and line 536 (`pixel_w=0.75` in the per-step log record). All other EMA wiring is identical to d001 (class at line 71, constants at 286–287, construction at 306–307, resume at 363–373, gated update at 494–495, dual-slot `save_ckpt()` at 415+, `ema_active` field at 531). `code/losses.py` and `code/inn_model.py` are byte-identical to parent runs/idea003/design003 — `pixel_w` is consumed via the existing `ImageSecretLoss` constructor parameter, no loss-code edit was needed.

**Training-signal sanity:**
- The masked L1 term is `pixel_w * mean(w · |stego - cover|)` with `w` shape `(B, 1, H, W)` broadcast over RGB; the mask is computed once per `forward` and reused across YUV and L1 (parent invariant — losses.py is unchanged so this is preserved by construction). Raising `pixel_w` to 0.75 only scales an already-correctly-wired term.
- L1 contribution scales linearly: `pixel_w * l_pixel` jumps from ~2.5e-3 to ~3.75e-3, ≈5× yuv_w·l_yuv — at the parent's §7-flagged edge but not over it. EMA's smoothing role on the (non-smooth-gradient) L1 term is the explicit hypothesis.
- EMA warmup, autocast separation, post-sched ordering, dual-slot ckpt all inherited from d001 unchanged.
- Bit-pathway weights (`secret_weight=args.bit_weight=20.0`) preserved — the stronger spatial term does not touch the bit-loss balance.
- Smoke test in test_output/metrics.jsonl ran 200 steps with `ema_active=1`, `pixel_w=0.75` correctly logged, `l_pixel ≈ 0.20` (consistent with d001's `l_pixel ≈ 0.20`, since the unweighted L1 distribution is the same — only the multiplier changed), no NaN/Inf, optimizer/scheduler progressing.

**Strongest objection:** Raising `pixel_w` while EMA is the only stabiliser risks `bit_acc_clean_min` regressing below the parent's already-marginal 0.9949 if EMA does not absorb the added L1-induced oscillation; the design explicitly flags this as the most plausible failure mode and instructs the implementation summary to record it. Ruled out as a blocking concern because the change is mechanically correct, the budget ratio is within the parent's 5× threshold, and the design specifies a regression-flag protocol rather than a silent failure.

**Fixes required:** none.
