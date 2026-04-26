# Code Review — design001

## design001
**Verdict:** APPROVED
**Automated gate:** PASS (review-check-implementation: structural PASS; check-scope PASS (parent baseline/, no infra/** edits); verify-claims PASS, 5 snippets checked.)
**Algorithm fidelity:** Implementation in `runs/idea002/design001/code/exp0_inn_train.py` matches `design.md` §4 exactly:
- `_DetDownUpResize` (lines 63–82) and `_HardChain` (lines 85–113) are the verbatim helpers from §4.1–§4.2; `_HardChain.forward` performs `nan_to_num → [-1,1]→[0,1] → resize → kornia JPEG → clamp → ×2-1`, matching `infra/exp0_inn_eval.py::_ResizeJPEGChain` semantics.
- `NoiseCurriculum.__init__` adds the three-chain bank `(0.5, 85), (0.5, 70), (0.3, 50)` (lines 151–157) and `_p_hard_max=0.4`, exactly as §4.3.
- `attack(step)` (lines 174–189) returns `None` when `α==0` (preserves clean phase), then uses `p_hard = _p_hard_max * α` to gate the hard-chain branch; on miss it scales `_attack.jpeg{1,2}.p` and `_attack.resize.p` by α — identical to baseline behavior. Branch label is recorded in `last_branch`.
- `to(device)` (lines 191–194) pushes both the AttackLayer and the hard-chain ModuleList.
- Startup sanity probe at lines 329–334 with NaN/Inf-poisoned fp16 input, as §4.4.
- Per-step `metrics.jsonl` includes `attack_branch=curriculum.last_branch` (line 491) and stdout line appends `br=<branch>` (line 501).
**Training-signal sanity:**
- Loss/optimizer/AMP/clip/cosine LR schedule untouched (verified by grep of `clip_grad_norm`, `AdamW`, `cosine`, `autocast` in the file — all match baseline).
- The hard-chain layer is differentiable end-to-end: `F.interpolate` is differentiable; `kornia.augmentation.RandomJPEG` is implemented as a differentiable approximation in kornia, so gradient flows from `bit_loss` back through the attack into the encoder. Confirmed by `test_output/metrics.jsonl` step 150 showing `attack_branch="hard[0.3,q50]"` with non-zero `bit_loss=0.139` and a non-trivial subsequent EMA update.
- Clean-phase invariant holds: at step 50 the test-output line shows `attack_branch="clean"` and `bit_acc=1.000`, confirming the clean phase is preserved before α ramps.
- `bit_weight=20.0` and pure-L1 host loss preserved (no `ImageSecretLoss` import).
- `test_output/ckpt_latest.pt` and `metrics.jsonl` present and well-formed.
**Strongest objection:** The hard chain unconditionally `clamp(0,1)` of the kornia-JPEG output may slightly bias gradient flow when JPEG output is itself in-range; however the same pattern is used in `infra/attacks.py` and the design explicitly mirrors it for fp16 safety — the trade-off is intentional and harmless for the eval-aligned regime this design targets. Ruled out.
