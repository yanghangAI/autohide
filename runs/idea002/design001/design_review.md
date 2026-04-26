## design001
**Verdict:** APPROVED

**Feasibility evidence:**
- Parent file `baseline/exp0_inn_train.py:72-135` defines `NoiseCurriculum` with `__init__`, `alpha`, `stage`, `attack`, `to` — exactly the surface design001 extends. The Bernoulli-mutation pattern at lines 122-131 (`self._attack.jpeg1.p = self._p_max["jpeg1"] * a`) is the same idiom the design proposes for the random branch.
- The training loop at line 372 (`attacked = attack_module(container) if attack_module is not None else container`) accepts any `nn.Module` returned by `curriculum.attack(step)`, so `_HardChain` instances (new `nn.Module`s) drop in cleanly.
- `infra/attacks.py:49-84` (`RandomDownUpResize`) provides the exact reference behavior the local `_DetDownUpResize` re-implements (bilinear down→up with `max(8, round(h*s))`, `align_corners=False`); fp32-cast + `nan_to_num` + `[-1,1]→[0,1]` sandwich at `infra/attacks.py:122-137` matches the `_HardChain.forward` exactly.
- `K.RandomJPEG(jpeg_quality=(q,q), p=1.0)` is supported (used in `infra/attacks.py:113-117`). The design correctly notes `kornia.augmentation as K` must be added to imports — `baseline/exp0_inn_train.py` does not currently import it (only `kornia.metrics` at line 165 inside `_run_validation`).
- Sanity-check probe pattern (line 262, `curriculum._attack.sanity_check()`) is the location the design specifies for inserting the new probe.

**Idea contradiction check:** Idea calls for "Eval-mirrored hard chains in training" with chain bank `{(0.5,70), (0.5,85), (0.3,50)}` and `p_hard(α)` ramp; design001 implements exactly that, including the gating `p_hard = 0.4 * α` so the clean phase (α=0) is preserved. No constraint violated: `bit_weight=20.0` retained, no `ImageSecretLoss`, no infra/ edits, locked CLI flags untouched.

**Strongest objection (ruled out):** Could the `0.4·α` hard-firing replace too much of the random distribution and cause forgetting of high-Q JPEG / mild-resize cases? Design explicitly retains `1 − p_hard(α)` of post-clean batches on the unchanged `AttackLayer`, and `p_hard_max = 0.4` keeps the random branch dominant (≥60%) even at α=1. The constraint also pins endpoints to the three eval chains and forbids extra points, so scope cannot drift.

**Fixes required:** None.
