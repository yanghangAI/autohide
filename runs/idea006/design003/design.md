**Design Description:** EMA stack (as design001) **plus** a single spatial-loss-axis perturbation: raise `pixel_w` from 0.5 to 0.75 to push more of `l_quality` onto the masked spatial L1 term. Tests whether EMA's averaging buys headroom for a stronger spatial reweighting than the un-EMA'd parent could safely sustain. All other knobs identical to design001 of this idea.

**Parent:** runs/idea003/design003

**Starting Point:** runs/idea003/design003

---

## 1. Goal and rationale

Per the idea's direction (3), probe whether EMA buys headroom on the **spatial loss axis** — i.e., whether the EMA-averaged eval weights tolerate a stronger masked-L1 contribution than the un-EMA'd parent. Of the two suggested perturbations (`pixel_w↑` vs `jnd_beta↑`), this design picks **`pixel_w↑` (0.5 → 0.75)** for the following reasons:

- The masked L1 term `pixel_w * mean(w · |stego - cover|)` is the **non-smooth** contributor to `l_quality` (L1 has discontinuous gradient at residual=0, unlike the YUV-MSE and FFL terms). It is therefore the term most likely to introduce late-training oscillations that EMA can profitably smooth. Raising `pixel_w` directly amplifies that signal.
- `jnd_beta=4.0` is already past the empirically validated point in `idea003/design001-003`; raising it further (e.g. 4 → 6) deepens textured-region down-weighting but does not change the smoothness of the loss landscape. The diagnostic "does EMA buy spatial headroom?" is sharper when the perturbed term is the non-smooth one.
- Per the parent's §7, `l_pixel * pixel_w ≈ 2.5e-3` at `pixel_w=0.5`, comparable to `yuv_w * l_yuv ≈ 7.5e-4`. Raising `pixel_w` to 0.75 gives `l_pixel * pixel_w ≈ 3.75e-3`, about 5× the YUV contribution but still well below the 5×-flag threshold the parent's §7 calls out (`l_pixel * pixel_w` exceeding `yuv_w * l_yuv` by more than 5×). This is a measured perturbation, not a regime change.

Hypothesis: vs design001 of this idea, this design produces a slightly higher PSNR (more spatial sparsity pressure → more residual energy localized to textured regions) at small or zero bit_acc cost (EMA absorbs the added late-training oscillation from the stronger L1 term). Vs the parent (no EMA), `pixel_w=0.75` alone might dip `bit_acc_clean_min` further; EMA absorbs that, so the EMA + stronger pixel_w is expected to dominate.

## 2. CLI flags / config for `baseline/exp0_inn_train.py`

All flags identical to design001 of this idea (and to the parent). **No locked flag changed.**

```
--num_blocks 16
--subnet_type rdb
--lr 2e-4
--img_weight 1.0
--bit_weight 20.0
--stage_a_steps 5000
--stage_b_steps 10000
--batch_size 16        # locked
--secret_len 896       # locked
--resolution 256       # locked
--max_steps 30000      # locked
--total_steps 30000    # locked
```

No new CLI flags. EMA decay (0.999) and EMA warmup (= `stage_a_steps = 5000`) are the same hard-coded constants as design001. `jnd_beta=4.0` and `jnd_mode="sobel"` unchanged from parent. **`pixel_w` changes from 0.5 (parent / design001 of this idea) to 0.75**, hard-coded at the `ImageSecretLoss(...)` constructor call site.

## 3. Files Builder may modify

- `code/exp0_inn_train.py`

Builder must **not** touch any of:
- `code/inn_model.py`
- `code/losses.py` (the parent's `ImageSecretLoss` already accepts `pixel_w` as a constructor argument; this design only changes the value passed at construction time, not the loss code)
- `code/train.py`
- `infra/**` (locked)

## 4. Algorithmic spec — exact changes to `code/exp0_inn_train.py`

Identical to design001 of this idea, §4.1 through §4.7 verbatim, **plus** the following one-line change to the `ImageSecretLoss(...)` constructor call:

```python
img_loss_mod = ImageSecretLoss(
    secret_weight=args.bit_weight,   # 20.0
    yuv_w=1.5,
    lpips_w=0.0,
    ffl_w=1.5,
    jnd_beta=4.0,
    pixel_w=0.75,                    # changed from 0.5
    jnd_mode="sobel",
).to(device)
```

In the per-step `metrics.jsonl` record, the existing `pixel_w=0.5` literal (inherited from the parent) must be updated to `pixel_w=0.75` so the logged constant matches what is actually used:

```python
pixel_w=0.75,
```

No other change vs design001 of this idea. The parent's `code/losses.py` is reused verbatim — only the **value** passed to its `pixel_w` parameter changes.

## 5. Expected behavior after the change

- `metrics.jsonl` per-step keys = parent keys ∪ `{ema_active}`. The `pixel_w` literal field reads `0.75`. The per-step `l_pixel` value distribution is the same shape as the parent's but the contribution `pixel_w * l_pixel` is 1.5× larger.
- `ckpt_latest.pt` payload keys = parent keys ∪ `{model_online, ema}`. `ckpt["model"]` holds EMA weights for `step >= 5000`.
- Eval `metrics.csv`: target `score >= 28.4`, ideally with PSNR slightly above design001 of this idea. Hard condition: `bit_acc_clean = 1.000` at every one of the 9 resolutions for the EMA weights at end of training.
- Wallclock: identical to design001 of this idea (the L1 term is a single elementwise abs+mul+mean, dominated by the existing FFL/YUV cost). Well within the 12 h SLURM budget.

## 6. Constraints and invariants Builder must preserve

1. **Locked CLI flags unchanged:** `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.
2. **Locked infra unchanged:** zero edits under `infra/**`.
3. **Single perturbation axis:** `pixel_w=0.75` is the *only* knob changed vs design001 of this idea. Builder must NOT also change `--lr`, `--num_blocks`, `--subnet_type`, `--img_weight`, `--bit_weight`, `--stage_a_steps`, `--stage_b_steps`, `jnd_beta`, `jnd_mode`, `yuv_w`, `lpips_w`, `ffl_w`, or `secret_weight`. Stacking perturbations destroys the diagnostic value of the comparison.
4. **No edits to `code/losses.py` or `code/inn_model.py`** — the parent's `ImageSecretLoss` already accepts `pixel_w`; only the constructor-call value changes.
5. **No new CLI flags.** `pixel_w` remains hard-coded at the constructor call site, exactly as in the parent.
6. **Hard condition:** `bit_acc_clean = 1.000` at every eval resolution must hold for the EMA weights at end of training.
7. EMA semantics, dual-slot ckpt, resume order, and autocast separation per design001 of this idea §4.
8. The mask `M` is computed exactly once per `forward` call and reused for both the YUV-weighted MSE and the L1 pixel term (inherited parent invariant — must not be regressed).
9. The L1 term operates in **RGB pixel space** (`(stego - cover).abs()`), not YUV space (inherited parent invariant).

## 7. Edge cases

- **Loss-balance check:** with `pixel_w=0.75` and typical `l_pixel ≈ 5e-3`, the L1 contribution to `l_quality` is `≈ 3.75e-3` vs `yuv_w * l_yuv ≈ 7.5e-4` and `ffl_w * l_ffl` of similar order. The L1 term is ≈ 5× the YUV term — at the **edge** of the parent's §7 5×-flag threshold. Builder must record the steady-state ratio `(pixel_w * l_pixel) / (yuv_w * l_yuv)` (averaged over steps 20000–30000) in the implementation summary. If it exceeds 6×, flag to Designer that `pixel_w=0.75` may be over the cliff for this dataset.
- **EMA warmup invariance:** EMA starts at step 5000, identical to design001 of this idea. The stronger pixel term means the late-training loss landscape is rougher, which is exactly the regime where EMA smoothing is hypothesised to help.
- **`bit_acc_clean_min` regression check:** if EMA weights at step 30000 give `bit_acc_clean_min < parent's 0.9949` at any resolution, flag as a regression. This is the most plausible failure mode for this design (stronger pixel L1 squeezes the bit-pathway budget; if EMA does not absorb the resulting oscillation, `bit_acc_clean_min` will drop).
- **AMP / autocast:** `(stego - cover).abs()` is fp16-safe; the mask multiplication broadcasts cleanly. No new fp16 issues vs parent.
- **Mask reuse with broadcast:** inherited from parent — `w` shape `(B, 1, H, W)`, `(stego - cover).abs()` shape `(B, 3, H, W)`, broadcast over RGB. Builder must not alter the parent's mask code.
