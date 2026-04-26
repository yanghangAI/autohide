# Code Review — design002

## design002
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS; check-scope PASS (parent baseline/, no infra/** edits); verify-claims PASS, 5 snippets checked.)
**Algorithm fidelity:** Implementation in `runs/idea002/design002/code/exp0_inn_train.py` matches `design.md` §4:
- `_sev_lo` / `_sev_hi` endpoints declared in `__init__` (lines 109–113) match `infra/attacks.py::SEVERITY` low/high values exactly (`jpeg_q_min=80↔40, chain_q_min=60↔20, resize_scale_min=0.9↔0.5, chain_p=0.0↔0.3`).
- `severity_alpha(step)` (lines 122–127) is a verbatim copy of `alpha(step)`, so α₂ tracks α₁ across the ramp window — matching the constraint in §6 ("α₂ schedule MUST be identical to α₁").
- `attack(step)` (lines 138–165) returns `None` when α₁==0 (clean preserved); otherwise linearly interpolates the four severity knobs by α₂ and writes `_attack.jpeg1.jpeg_quality`, `_attack.jpeg2.jpeg_quality`, `_attack.resize.scale_min`, then scales firing probabilities by α₁ as required.
- `stage(step)` updated to `ramp(a1=...,a2=...)` (lines 129–136) — confirmed in `test_output/metrics.jsonl` ("stage": "ramp(a1=0.49,a2=0.49)" at step 100).
- Startup sanity probe added in main (probes ramp_start+1, midpoint, ramp_end+1).
- Per-step `metrics.jsonl` includes `severity_alpha`, `jpeg1_q_min`, `resize_scale_min` — confirmed in test_output (`severity_alpha=0.49, jpeg1_q_min=60.4, resize_scale_min=0.704` at step 100; `1.0, 40.0, 0.5` at step 200).
**Training-signal sanity:**
- Loss/optimizer/clip/cosine LR/AMP unchanged.
- Severity ramp is monotone and produces the expected tuple values: at α₂=0.49, `jpeg1_q_min ≈ 80 + 0.49*(40-80) = 60.4`, `resize_scale_min ≈ 0.9 + 0.49*(0.5-0.9) = 0.704` — both match the metrics.jsonl trace exactly.
- The mutation path uses Python-tuple reassignment to `RandomJPEG.jpeg_quality`; the design explicitly required Builder to verify kornia honors this (§4.3 note). The startup probe ran without exception (`[init] severity-annealing sanity check passed.` in slurm log) AND the metrics show jpeg1_q_min=40.0 at step 200, confirming the new bounds are propagated to the layer's behavior. Rebuild fallback was not needed.
- Clean phase invariant: at step 50 the slurm log shows `clean α=0.00 bit_acc=1.000 sev_α=0.00` — preserved.
- `bit_weight=20.0` and pure-L1 loss retained.
- `test_output/ckpt_latest.pt` and `metrics.jsonl` present and well-formed.
**Strongest objection:** The constraint in §6 says "low/high endpoints MUST equal `infra/attacks.py::SEVERITY` at the time of writing"; if `SEVERITY` is later changed in infra (it is locked, but worth noting), the duplicated constants here will silently drift. This is a maintenance risk, not a correctness issue for this run, and the design explicitly authorizes the duplication because infra is locked. Ruled out.
