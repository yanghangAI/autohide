## design002
**Verdict:** APPROVED

**Feasibility evidence:**
- `baseline/exp0_inn_train.py:371` — `container = container.clamp(-1.0, 1.0)` exists exactly as quoted; one-line replacement with `ste_clamp(container, -1.0, 1.0)` is mechanically valid.
- `baseline/exp0_inn_train.py:158` — `container = model.embed(cover, secret).clamp(-1, 1)` inside `_run_validation` (which is `@torch.no_grad()` per line 139); design correctly leaves this untouched.
- `baseline/exp0_inn_train.py:62` (`def _psnr`) and `baseline/exp0_inn_train.py:72` (`class NoiseCurriculum`) — insertion point for `ste_clamp` helper between them is valid.
- `baseline/exp0_inn_train.py:387-393` — `with torch.no_grad():` diagnostics block computes `container_f = container.float()`; the `sat_frac` field can be appended to the existing `rec` dict at line 413 using the same `container_f` tensor without re-allocation.
- STE identity `x + (x.clamp(lo,hi) - x).detach()` is differentiable — autograd sees only the leading `x` (gradient = 1 everywhere); the `.detach()` correctly severs the graph for the constant offset. INN coupling shape is preserved: `ste_clamp` is elementwise and shape-preserving.

**Idea contradiction check:** `idea.md` direction 2 prescribes "Replace the hard `container.clamp(-1.0, 1.0)` with a straight-through estimator that preserves the forward clip but passes the gradient through. No EMA. Isolates the gradient-flow effect from the weight-averaging effect." Design002 implements exactly this. No contradiction.

**Strongest objection (ruled out):** that under AMP/autocast the `(x.clamp - x).detach()` expression could trigger a graph rebuild every step or introduce dtype mismatch. Ruled out: the operation is composed of standard differentiable primitives that autocast handles natively (clamp, sub, add); `.detach()` is recognized by autograd as a constant-graph operation, and the design explicitly flags the walltime sanity-check (`> 1.1× baseline`) as the empirical guard if a rebuild were occurring. Forward values are byte-identical at fp32 (within fp16 rounding under AMP), so `bit_acc_clean = 1.000` is preserved by construction since eval uses the locked `infra/exp0_inn_eval.py` hard-clamp path.

**Locked-scope check:** PASS. No `infra/**` edits. CLI flags identical to baseline. Eval attack suite untouched. Validation-path clamp explicitly preserved.
