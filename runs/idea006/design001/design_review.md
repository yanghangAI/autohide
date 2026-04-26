## design001
**Verdict:** APPROVED

**Feasibility evidence:**
- Parent train script `runs/idea003/design003/code/exp0_inn_train.py:63` defines `_psnr` and `:73` defines `class NoiseCurriculum` — the WeightEMA class can be inserted between them as specified in §4.1.
- `:218,:220` declare `--stage_a_steps` and `--stage_b_steps` argparse flags, so `args.stage_a_steps` (used as `EMA_WARMUP_STEPS` in §4.2) resolves correctly.
- `:249-251` print the `[model] INNCodec n_params=...` line — §4.3 anchors EMA construction immediately after.
- `:305-307` show `if ckpt_path.exists(): ... model.load_state_dict(ck["model"])` — §4.4's dual-slot replacement is mechanically possible.
- `:414` `sched.step()` in the training loop is the anchor point for §4.5's `ema.update(model)` insertion.
- `:348` defines `save_ckpt()` for §4.6's body replacement.
- `:442-455` `rec = dict(...)` includes `pixel_w=0.5`, `jnd_beta=4.0`, `jnd_mode="sobel"` — adding `ema_active` (§4.7) is a one-line dict augmentation.
- `runs/idea004/design001/code/exp0_inn_train.py:70` (`class WeightEMA`), `:295-296`, `:315-316`, `:357-367`, `:415-428`, `:480-481`, `:517` confirm the EMA pattern is byte-identical to what design001 prescribes.
- `runs/idea003/design003/code/losses.py:88-160` confirms `ImageSecretLoss` already accepts `pixel_w`, `jnd_beta`, `jnd_mode` constructor args (§6.8 invariant holds).

**Idea contradiction check:** Idea direction (1) prescribes a "pure composition" — d001 verbatim mirrors `idea003/design003`'s flags and adds the `idea004/design001` EMA constants (decay=0.999, warmup=stage_a_steps). No mechanism, dataset, or stated goal is reversed. Locked CLI flags untouched; no `infra/**` edits prescribed; LPIPS/STE/HEM exclusions honored.

**Strongest objection:** The parent's `bit_acc_clean_min=0.9949` slightly violates the design's stated hard condition `bit_acc_clean = 1.000`. Since EMA inherits the online weights at warmup (step 5000, post-clean phase), the clean pathway is structurally preserved — the dip is observed at later steps after attack ramp begins. The §6.5 mitigation language correctly acknowledges this is "a structural mitigation" rather than a guarantee, and §7 instructs Builder to flag any regression. This is the correct treatment of a soft signal vs a hard gate, not a contradiction.
