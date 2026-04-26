## design001
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS, check-scope PASS vs parent runs/idea003/design003, verify-claims PASS on 7 snippets)
**Algorithm fidelity:** Compared against design.md §4.1–§4.7. Verified in code/exp0_inn_train.py:
- `WeightEMA` class at line 71 matches §4.1 verbatim (fp32 shadow, `mul_(d).add_(p.detach().float(), alpha=1.0 - d)`, buffers copied, state_dict/load_state_dict implemented).
- EMA constants at lines 286–287: `EMA_DECAY = 0.999`, `EMA_WARMUP_STEPS = args.stage_a_steps` — matches §4.2.
- EMA construction at lines 306–307 — matches §4.3.
- `ImageSecretLoss(...)` constructor at lines 314–322 unchanged from parent (`secret_weight=args.bit_weight, yuv_w=1.5, lpips_w=0.0, ffl_w=1.5, jnd_beta=4.0, pixel_w=0.5, jnd_mode="sobel"`) per §6.8.
- Dual-slot resume at lines 363–373 matches §4.4 exactly (online slot loaded first, then EMA, with the documented fallback path).
- `save_ckpt()` at lines 415+ writes EMA shadow into `ckpt["model"]` post-warmup with the online-state union for non-shadowed keys, and stores `model_online` + `ema` slots — matches §4.6.
- EMA update at lines 494–495 placed after `sched.step()` (line 492) and before the `with torch.no_grad()` diagnostics block, gated by warmup — matches §4.5.
- `ema_active=int(global_step >= EMA_WARMUP_STEPS)` field at line 531 — matches §4.7.
- `code/losses.py` and `code/inn_model.py` are byte-identical to parent runs/idea003/design003 (`diff -q` returns no output), satisfying the "no edits to losses/inn_model" constraint.

**Training-signal sanity:**
- EMA update is outside `torch.amp.autocast` and reads `p.detach().float()` → fp32 shadow, no autocast contamination (matches §6.7).
- EMA update is gated on `global_step >= EMA_WARMUP_STEPS` and runs *after* `optim.step()` and `sched.step()`, so the shadow tracks post-update online weights — correct.
- Dual-slot resume order is online-then-EMA, so an interrupted run re-instantiates EMA before any update fires; the fallback `WeightEMA(model, ...)` re-initialises from current online weights — safe.
- `save_ckpt()` writes EMA into the eval-consumed `ckpt["model"]` slot, so locked `infra/exp0_inn_eval.py` (which reads `ckpt["model"]`) sees EMA weights at end of training without any infra change. Buffer-set is unioned from `model.state_dict()` so non-trainable buffers (e.g. BN stats if any) are preserved.
- Loss/optimizer/curriculum unchanged from parent; the JND+L1 mask path (Sobel mode, β=4.0, pixel_w=0.5) is inherited verbatim. Bit-pathway (`bit_weight=20.0`) is preserved.
- Smoke test in test_output/metrics.jsonl ran 200 steps with `ema_active=1` from step 50 onward (smoke harness sets `stage_a_steps` very low), `pixel_w=0.5` literal logged, no NaN, optimizer/scheduler progressing (lr ramps then decays to 0).

**Strongest objection:** EMA buffers (model.named_buffers) are copied verbatim each update with `.copy_(b.detach())` rather than smoothed. For models with running BN statistics this would defeat the purpose of EMA, but the INN here uses RDB subnets without BN, so buffer copy is effectively a no-op aside from the integer-dtype curriculum buffers; this matches the proven idea004/design001 pattern. Ruled out as a real concern.

**Fixes required:** none.
