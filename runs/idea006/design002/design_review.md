## design002
**Verdict:** APPROVED

**Feasibility evidence:**
- All design001-of-this-idea anchors apply (see d001 review). The d002 delta is a single CLI value `--stage_b_steps 14000`.
- `runs/idea003/design003/code/exp0_inn_train.py:220` registers `--stage_b_steps` as an existing int argparse flag (default 10000) — passing 14000 is a flag-value change, not a code change.
- `:97-99` `NoiseCurriculum.__init__` consumes `stage_a_steps` + `stage_b_steps` to compute `ramp_start = stage_a_steps`, `ramp_end = stage_a_steps + stage_b_steps`. With `5000 + 14000 = 19000 < 30000 (max_steps)` the ramp completes inside the run; no hard-coded 15000 cap exists in the curriculum.
- `:270-271` instantiate `NoiseCurriculum(stage_a_steps=args.stage_a_steps, stage_b_steps=args.stage_b_steps, ...)` — the new value is consumed correctly.
- All §4.1–§4.7 EMA anchors identical to d001, all confirmed against parent + `idea004/design001` source.

**Idea contradiction check:** Idea direction (2) authorizes exactly one of `{lr↑, stage_b_steps↑}`; d002 explicitly picks `stage_b_steps↑` and justifies the choice (AMP/clip stability under 16-block RDB). `--stage_a_steps` held at 5000 preserves EMA-warmup parity with d001, satisfying the diagnostic-isolation constraint. Locked CLI flags (batch_size, secret_len, resolution, max_steps, total_steps) untouched. No `infra/**` edits.

**Strongest objection:** Lengthening `stage_b_steps` to 14000 leaves only `30000 - 19000 = 11000` steps in the post-ramp tail (vs 15000 in the parent), reducing the "stable α=1.0" optimization window by ~27%. This could bite the bit pathway's late convergence regardless of EMA. The design correctly flags this in §7 (curriculum boundary check) and the EMA is hypothesised to absorb the additional late oscillation. Acceptable as a tested perturbation, not a structural break.
