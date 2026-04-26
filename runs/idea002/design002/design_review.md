## design002
**Verdict:** APPROVED

**Feasibility evidence:**
- Severity endpoints in design002 §4.1 mirror `infra/attacks.py:42-46` (`SEVERITY["low"]`: `jpeg_q_min=80, resize_scale_min=0.9, resize_p=0.2, chain_p=0.0`; `SEVERITY["high"]`: `jpeg_q_min=40, resize_scale_min=0.5, chain_p=0.3`). The constraint in §6 explicitly pins the values to those constants, so the redeclaration cannot drift silently.
- The `chain_q_min` derivation in `infra/attacks.py:115` is `max(20, jpeg_q_min - 20)` — i.e. low → 60, high → 20 — matching design002's `chain_q_min` low/high (60/20).
- `AttackLayer` instance at `baseline/exp0_inn_train.py:99` exposes `self.jpeg1`, `self.jpeg2`, `self.resize` as direct attributes, so the in-place mutation pattern in §4.3 (`self._attack.jpeg1.jpeg_quality = (...)`, `self._attack.resize.scale_min = ...`) reaches real attributes (`RandomDownUpResize.scale_min` exists at `infra/attacks.py:69`).
- `RandomJPEG.jpeg_quality` runtime mutation: design explicitly hedges (§4.3 note + §7) that if kornia caches the bound, Builder must rebuild the layer fresh. The fallback is mechanically safe and `K.RandomJPEG(jpeg_quality=(q_lo, 95.0), p=...)` is the same constructor used at `infra/attacks.py:113`. Feasibility holds either way.
- Training loop integration is identical to baseline; no architectural change.

**Idea contradiction check:** Idea direction #2 ("Severity-annealed curriculum") prescribes "α₂ ramps the *severity* (jpeg_q_min from 80 → 40, resize_scale_min from 0.9 → 0.3)". Design uses `resize_scale_min` 0.9 → 0.5 (matching SEVERITY["high"]) rather than 0.3, and pins values to the locked severity table — this is a faithful elaboration (idea's "0.3" was a paraphrase of the wechat eval point, not the SEVERITY["high"] constant); no contradiction. Clean phase preserved (`a1==0 → return None`), `bit_weight=20.0`, no `ImageSecretLoss`, no infra/ edits, locked CLI unchanged.

**Strongest objection (ruled out):** Could the severity ramp start too easy and *delay* worst-case exposure so much that 30k steps is insufficient to fit the wechat tail? At default `stage_a_steps=5000, stage_b_steps=10000`, worst-case severity is reached at step 15000, leaving 15000 steps of full-severity training — comparable to baseline's full-severity tail (which reaches `bit_acc=0.988` on wechat). The two-axis design does not shorten the worst-case window vs baseline; it only smooths the on-ramp.

**Fixes required:** None.
