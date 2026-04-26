## design003
**Verdict:** APPROVED

**Feasibility evidence:**
- All d001-of-this-idea EMA anchors apply (see d001 review).
- `runs/idea003/design003/code/losses.py:96-111` shows `ImageSecretLoss.__init__` accepts `pixel_w: float = 0.0` and stores it as `self.pixel_w`. `:137-160` use `self.pixel_w` to weight the masked L1 term — passing `pixel_w=0.75` at construction time changes the value with no loss-code edit.
- `runs/idea003/design003/code/exp0_inn_train.py:258-265` shows the parent's `ImageSecretLoss(...)` constructor call already includes `pixel_w=0.5, jnd_mode="sobel"` — the design's one-line value change at `:264` is mechanically trivial.
- `:454` shows the per-step `rec` dict has `pixel_w=0.5` literal — updating to `0.75` keeps the logged constant in sync with the constructor argument (no semantic change to the metric).
- `code/losses.py` is correctly listed as immutable (§6.4) since only the constructor argument value changes.

**Idea contradiction check:** Idea direction (3) authorizes exactly one of `{pixel_w↑, jnd_beta↑}`; d003 picks `pixel_w↑ (0.5 → 0.75)` and justifies it (L1 is the non-smooth term most likely to benefit from EMA smoothing). All other knobs identical to d001 of this idea. Locked CLI flags untouched, no `infra/**` edits, LPIPS/STE/HEM exclusions honored. The parent's mask-reuse and RGB-pixel-space invariants (§6.8, §6.9) are explicitly preserved.

**Strongest objection:** The design's §7 self-flag notes that with `pixel_w=0.75`, the L1 contribution `(pixel_w * l_pixel) ≈ 3.75e-3` reaches ~5× the YUV contribution — at the edge of the parent's documented 5×-flag threshold. If steady-state ratio exceeds 6× the bit pathway may collapse. The design correctly instructs Builder to record this ratio over steps 20000–30000 and flag if it crosses 6× — this is the appropriate mitigation (measure, do not silently tune). Acceptable as a measured single-axis perturbation.
