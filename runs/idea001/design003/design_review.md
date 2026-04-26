## design003
**Verdict:** APPROVED

**Feasibility evidence:**
- `baseline/losses.py:61` is exactly `l_lpips = self.lpips(stego, cover).mean()  # inputs already in [-1, 1]` — the literal line the design wraps with the `if self.lpips_w == 0.0:` short-circuit (§4.1). Insertion is mechanical and additive.
- `baseline/losses.py:46` stores `self.lpips_w = lpips_w`, so `self.lpips_w == 0.0` is a valid attribute check.
- `baseline/losses.py:48` constructs `self.lpips = lpips_lib.LPIPS(net="alex", verbose=False)` unconditionally; the design correctly notes this stays as-is and just sits idle when `lpips_w=0`.
- `l_lpips` is consumed only by `l_quality = … + self.lpips_w * l_lpips + …` on line 65; substituting a `torch.zeros((), device=stego.device, dtype=stego.dtype)` keeps `l_quality` and `total` differentiable (zero contribution × zero weight) and the `logs["l_lpips"]` entry remains a tensor for `.detach()` on line 73.
- Train-loop wiring is identical to design001/002 and rests on the same lines (`baseline/exp0_inn_train.py:246` insertion point; `:366-377` L1+BCE replacement; `:371` clamp; `:395` precedent for nested `autocast(enabled=False)`).

**Idea contradiction check:** `idea.md` direction (3) is "YUV+FFL only (no LPIPS) — cheaper, no extra forward pass through AlexNet." Design003 sets `lpips_w=0.0` and adds a guard so the AlexNet forward is actually skipped — exact elaboration, no contradiction.

**Strongest objection (ruled out):** Editing `baseline/losses.py` (a non-`infra/` file, but still parent-shared) could in principle change behavior for designs that import `ImageSecretLoss` with `lpips_w > 0`. Ruled out because the guard is strictly conditional on `self.lpips_w == 0.0`; for any nonzero weight the `else` branch runs the original line verbatim, so design001/002 (and any future caller) are byte-equivalent. Edits live under `code/baseline/losses.py` (the design's own copy), per §2 — not a cross-design mutation.

**Fixes required:** N/A (APPROVED).
