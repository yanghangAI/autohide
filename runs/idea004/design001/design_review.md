## design001
**Verdict:** APPROVED

**Feasibility evidence:**
- `baseline/exp0_inn_train.py:233` — `args = ap.parse_args()` exists; constants `EMA_DECAY` / `EMA_WARMUP_STEPS` can be inserted directly after.
- `baseline/exp0_inn_train.py:249` — `print(f"[model] INNCodec n_params=...")` line exists; `WeightEMA(model, decay=EMA_DECAY)` insertion point is valid.
- `baseline/exp0_inn_train.py:289-291` — `if ckpt_path.exists():` block with `model.load_state_dict(ck["model"])` exists; resume modification target is valid.
- `baseline/exp0_inn_train.py:332-351` — `save_ckpt()` body to be replaced; `payload` dict has `"model": model.state_dict()` slot — substituting EMA shadow into that slot is mechanically possible.
- `baseline/exp0_inn_train.py:385` — `sched.step()` followed by `with torch.no_grad():` diagnostics at line 387; EMA update insertion between them is feasible.
- `baseline/exp0_inn_train.py:413` — `rec = dict(step=global_step, stage=stage_name, alpha=attack_alpha, ...)` exists; adding `ema_active` field is straightforward.
- INN model exposes `.named_parameters()` / `.named_buffers()` (standard nn.Module); no architecture-specific hook needed.

**Idea contradiction check:** `idea.md` direction 1 prescribes EMA-only with single fixed decay (recommends 0.999), warmup after the clean phase (`step >= stage_a_steps`), saved into the eval-consumed checkpoint slot, with online weights kept separately for resume. Design001 implements exactly this — decay 0.999, warmup at `args.stage_a_steps`, dual-slot `model` / `model_online` payload. No contradiction.

**Strongest objection (ruled out):** that writing EMA weights to `ckpt["model"]` could violate the locked-infra contract. Ruled out: `infra/exp0_inn_eval.py` is an opaque consumer of `ckpt["model"]` as a state_dict; the contract is "this slot is the model state_dict at eval time." Substituting an EMA-averaged state_dict (same architecture, same key names, same shapes) honors the contract — the design does not edit infra and does not add/remove eval-side keys. The `bit_acc_clean = 1.000` hard condition is the empirical safeguard if EMA weights ever degrade clean decoding, with the warmup-after-stage_a mitigation as documented.

**Locked-scope check:** PASS. No `infra/**` edits. CLI flags identical to baseline (no locked flag changed). Eval attack suite untouched (locked `infra/exp0_inn_eval.py`).
