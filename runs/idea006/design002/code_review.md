## design002
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS, check-scope PASS vs parent runs/idea003/design003, verify-claims PASS on 7 snippets)
**Algorithm fidelity:** d002's design.md §4 mandates the d001 EMA edits verbatim, with the only delta being the CLI flag value `--stage_b_steps 14000`. Confirmed via `diff` that `code/exp0_inn_train.py` is byte-identical to d001's (the curriculum value lives in the SLURM submit script / CLI invocation, not in the source). All d001 code-fidelity checks therefore apply: `WeightEMA` class, EMA constants (lines 286–287), construction (306–307), dual-slot resume (363–373), gated `ema.update()` (494–495), dual-slot `save_ckpt()` (415+), and `ema_active` log field (531). `ImageSecretLoss(...)` constructor at lines 314–322 unchanged from parent (pixel_w=0.5, jnd_beta=4.0, jnd_mode="sobel"). `code/losses.py` and `code/inn_model.py` are byte-identical to parent runs/idea003/design003.

**Training-signal sanity:**
- Curriculum boundary check (per design §7): `stage_a_steps=5000 + stage_b_steps=14000 = 19000`; tail = 30000 − 19000 = 11000 steps. The existing `NoiseCurriculum` consumes `stage_b_steps` as a parameter and α reaches 1.0 inside that window — no hard-coded 15000 assumption was found; the curriculum module's α schedule is parameterised purely on (stage_a_steps, stage_b_steps).
- EMA warmup remains at step 5000 — identical timing to d001, so the EMA-vs-no-EMA ablation against d001 is clean.
- All EMA training-signal checks from d001 carry over (autocast separation, post-sched.step ordering, dual-slot ckpt, eval-slot semantics).
- Smoke test in test_output/metrics.jsonl ran 200 steps with `ema_active=1`, `pixel_w=0.5` (correct — d002 does NOT change pixel_w), losses progressing normally.

**Strongest objection:** Lengthening `stage_b_steps` shortens the high-α tail (11000 vs parent's 15000 steps), which could under-converge the bit pathway at full attack severity. Ruled out as blocking because `--max_steps` and `--total_steps` remain locked at 30000 and the parent's bit_acc curve plateaus well before step 25000, so 11000 high-α steps are sufficient; the trade is intentional and motivated in the design.

**Fixes required:** none.
