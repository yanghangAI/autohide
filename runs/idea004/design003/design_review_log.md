## design003
**Verdict:** APPROVED

**Feasibility evidence:**
- All line-number references inherited from design001 + design002 verified against `baseline/exp0_inn_train.py`:
  - `:62` `def _psnr`, `:72` `class NoiseCurriculum` — insertion zone for `WeightEMA` and `ste_clamp` helpers.
  - `:158` validation-path clamp (preserved as hard clamp, correctly).
  - `:233` `args = ap.parse_args()` — constants insertion point.
  - `:249` `print(f"[model] INNCodec n_params=...")` — EMA construction insertion point.
  - `:289-291` resume block with `model.load_state_dict(ck["model"])` — replaceable with dual-slot logic.
  - `:332-351` `save_ckpt()` body — replaceable with dual-slot payload.
  - `:371` `container = container.clamp(-1.0, 1.0)` — one-line STE substitution.
  - `:385` `sched.step()` — EMA update insertion before line 387 `with torch.no_grad():` diagnostics block.
  - `:413` `rec = dict(...)` — `ema_active` and `sat_frac` field additions valid.
- The two interventions touch disjoint code regions: EMA modifies the optimizer/checkpoint side (post `sched.step`, save/resume), STE modifies the forward graph (line 371). No conflicting edits at any line.

**Idea contradiction check:** `idea.md` direction 3 prescribes "EMA + STE-clip combined. Both interventions applied. If the two effects are independent and additive, this should be the strongest configuration." Design003 is the literal union of design001 and design002 with identical EMA decay (0.999) and identical warmup (`stage_a_steps`), explicitly so per-axis decomposition is meaningful. No contradiction.

**Strongest objection (ruled out):** that the STE-induced trajectory shift could change the EMA shadow in a way that breaks `bit_acc_clean = 1.000`. Ruled out: STE only affects gradient flow on saturated pixels; forward container values are byte-identical to baseline at fp32. The clean phase (`step < stage_a_steps`) is when `bit_acc_clean = 1.000` is established, and EMA does not update until *after* that phase (warmup tied to `stage_a_steps`). EMA then averages the post-clean trajectory; the worst-case is EMA tracking the online (STE-modified) weights, which themselves have the hard-condition empirical safeguard at eval. Design correctly notes that builder must not preemptively clip if `sat_frac` runaway occurs — it is a finding, not a bug.

**Locked-scope check:** PASS. No `infra/**` edits. CLI flags identical to baseline. Eval attack suite untouched.

