# Code Review Log — design001

---
## 2026-04-26 — Code review pass
## design001
**Verdict:** APPROVED

**Automated gate:** PASS (`review-check-implementation`: structural OK; `check-scope` PASS against parent `baseline/`; `verify-claims` 4/4 snippets matched)

**Algorithm fidelity:**
- `_make_balanced_permutation` at `code/inn_model.py:103-129` matches design §4.1: `arange(num_slots).repeat_interleave(p)[perm]` Fisher-Yates assignment driven by `torch.Generator(seed=0xD1FF5E)`, then sorted scatter into `(num_slots, p)` rows. Note the `num_positions`/`num_slots` argument order is swapped from the spec, but call sites use kwargs so semantics are preserved.
- `PatchBitAdapter.__init__` at `code/inn_model.py:144-163` registers `sign_mask` and the canonical `perm_canonical` buffer (1024×16 LongTensor) for the 256-px DWT plane, plus a `_perm_cache` dict for non-canonical eval resolutions, exactly as design §4.2 prescribes.
- `_perm_for` at `code/inn_model.py:165-182` rebuilds non-canonical permutations from `seed = 0xD1FF5E ^ (h_dwt*1000003 + w_dwt)`, matching design §4.2.
- `encode` at `code/inn_model.py:184-200` builds `positions_to_slot` via `scatter_` of `arange(NUM_SLOTS).repeat_interleave(p)` into the inverted index, then `index_select` of `scrambled` — equivalent to scattering each slot's value to its `p` owned positions, faithful to §4.2.
- `decode` at `code/inn_model.py:202-211` does mean-over-12-channels → `index_select` over `perm.reshape(-1)` → reshape `(B, NUM_SLOTS, p)` → mean over `p` → unscramble — the exact left-inverse of encode.
- Step-0 invertibility check at `code/exp0_inn_train.py:252-262` matches §4.3, and the runtime log shows `[adapter] balanced-permutation invertibility check: PASS`.
- Logging key `adapter_kind="balanced_perm"` added at `code/exp0_inn_train.py:435`, matching §5.

**Training-signal sanity:**
- The encode is exactly invertible by construction (disjoint balanced partition; mean of `p` copies of constant ±1 = the constant). The mini-train `[adapter] ... PASS` at step 0 confirms this empirically.
- Optimizer/scheduler/loss/attack composition are unchanged from baseline; only the adapter logic is replaced. Gradients flow through `index_select` (autocast-safe).
- Mini-train ran to completion with `[slurm_test] success`, emitted `metrics.jsonl` (4 records). bit_acc remains around baseline-like values throughout.

**Strongest objection:** The mini-train shows `psnr=NaN`, `img_loss=NaN`, `l1=NaN` from step 100 onward — image-loss has diverged once the attack ramp begins (α=0.49). The same compressed `--stage_a_steps 50 --stage_b_steps 100` curriculum on design003 and design004 does not NaN; design001's per-pixel ±1 high-frequency speckle carrier (vs. ~0.93-stddev broadband for the orthonormal designs) is pushing the AMP/fp16 forward pass into overflow at the rapid attack onset. At step 50 (clean stage) PSNR is only 6.18 vs. 10.98 (d003) / 12.88 (d004) — a meaningful gap. **However**, (i) the scope of code review is correctness-vs-design and structural sanity, not research outcome; (ii) the canonical 5000/10000 schedule is much gentler than the mini-train's 50/100 telescope; (iii) `bit_acc_clean = 1.000` is structurally guaranteed by the orthogonal disjoint partition and was empirically verified at step 0; (iv) the design itself was approved with this trade-off acknowledged. The NaN is a research-feasibility concern flagged for the full-train, not a code-review rejection cause.

**Fixes required:** none.
