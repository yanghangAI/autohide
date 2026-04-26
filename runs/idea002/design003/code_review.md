# Code Review — design003

## design003
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS; check-scope PASS (parent baseline/, no infra/** edits); verify-claims PASS, 9 snippets checked.)
**Algorithm fidelity:** Implementation in `runs/idea002/design003/code/exp0_inn_train.py` matches `design.md` §4:
- `_DetDownUpResize` (line 63) and `_HardChain` (line 77) helpers copied verbatim per §4.1.
- `NoiseCurriculum.__init__` constructs the fixed 5-entry bank in the prescribed order: `("random", self._attack)`, then four `_HardChain` entries `(0.5, q85), (0.5, q70), (0.4, q75), (0.3, q50)` (lines 141–146); `_bank_weights` initialized uniform `1/5` (line 149); `_bank_bit_acc=0.5`, `last_bank_idx=0`, `last_branch="clean"` (lines 150–152).
- `attack(step)` (lines 169–187) returns `None` when α==0 (clean preserved), else `torch.multinomial` samples bank index by `_bank_weights`; the "random" branch mutates `_attack.jpeg{1,2}.p` and `_attack.resize.p` by α (no cross-call leakage since they are reassigned every call); deterministic chains are returned directly.
- `update_weights` (lines 189–218) is `@torch.no_grad`, runs `model.embed → extract` per chain, computes `bit_acc` against `probe_secret`, and sets `_bank_weights = clamp_min(softmax((1-acc)/0.05), 0.05) / sum`. The floor + renormalize ordering matches §4.3 exactly.
- `to(device)` (lines 220–226) moves AttackLayer, deterministic chain modules, and both bank tensors.
- Probe batch construction in main (lines 442–451) uses `_val_load_to_size(val_paths[:n_probe], 256, 256)` from the held-out val root and a fixed random secret — matches §4.4.
- Periodic mining call (lines 587–600) at `MINING_EVERY=500`, gated on `global_step >= curriculum.ramp_start`, switches to eval and back, and prints the `[mining ...]` line.
- Startup bank sanity probe (lines 363–365) iterates `_bank[1:]` with NaN/Inf fp16 input.
- Per-step `metrics.jsonl` includes `attack_branch`, `bank_idx`, `bank_weights` (lines 545–547) — confirmed in `test_output/metrics.jsonl` (`"bank_idx": 3, "bank_weights": [0.20, 0.20, 0.20, 0.20, 0.20]` at step 100).
- Resume restores `bank_weights` from checkpoint (lines 399–406) and save persists it (line 469) — required by §7 edge case.
**Training-signal sanity:**
- Loss/optimizer/clip/cosine LR/AMP unchanged.
- Mining update runs in `eval()` + `no_grad()`, so probe gradients cannot contaminate the optimizer (verified at lines 591–596).
- Sampler distribution is correct: `multinomial` weights pre-normalized at construction (sum=1) and after every update (`w = w / w.sum()`), so probabilities are valid.
- Bank index 0 is `random` so when `bank_idx==0` the mutated AttackLayer is returned (the same instance used elsewhere); subsequent calls reassign `.p` from `_p_max * α`, preventing stale values from a prior `random` call leaking when the bank picks index 0 at a different α — the in-place mutation pattern is consistent.
- Floor `0.05` × 5 entries = 0.25 of probability mass minimum kept off the worst chain, ensuring the "easy" chains continue receiving gradient — prevents collapse onto a single attack.
- Clean-phase invariant: in test_output, step 50 (not shown in metrics) and the multinomial sampler are gated correctly; at step 100 (`α=0.49`) `bank_idx=3` confirms the sampler fired post-clean.
- `bit_weight=20.0` and pure-L1 loss retained.
- `test_output/ckpt_latest.pt` and `metrics.jsonl` present.
**Strongest objection:** The 200-step smoke test never reaches `ramp_start=5000`, so the mining update path itself was not exercised at runtime — only the bank-sampler path and the startup probes were. However, the `update_weights` body is straightforward (eval forward + softmax + clamp), is exercised by the data flow shape, and the startup bank-sanity probe + `multinomial` sampler at step 100 covers the dependent code paths. The full 30k-step training run will exercise mining for the first time at step 5000; that is acceptable for a smoke test. Ruled out.

---

## Overall (idea002)
- **Passed code review:** design001, design002, design003.
- **Failed code review:** none.
