**Design Description:** EMA stack (as design001) **plus** a single optimization-axis perturbation: lengthen `--stage_b_steps` from 10000 to 14000 to give the bit pathway more attack-conditioning time before the long ramp tail. Tests whether EMA's late-training smoothing tolerates a curriculum perturbation that the un-EMA'd parent might not. All other knobs identical to design001 of this idea.

**Parent:** runs/idea003/design003

**Starting Point:** runs/idea003/design003

---

## 1. Goal and rationale

Per the idea's direction (2), probe whether EMA buys *headroom* on the optimization axis — i.e., whether the EMA-averaged eval weights tolerate a more aggressive curriculum setting that would otherwise add late-training oscillation. Of the two suggested perturbations (`lr↑` vs `stage_b_steps↑`), this design picks **`stage_b_steps↑`** for the following reasons:

- The parent (`runs/idea003/design003`) already runs at `--lr 2e-4` and `--num_blocks 16 --subnet_type rdb`, a relatively heavy model. Raising `lr` above 2e-4 with a 16-block RDB INN risks AMP-scale-overflow and gradient-clip saturation regardless of EMA, which would conflate "EMA buys headroom" with "lr was past the stability cliff." The diagnostic is cleaner with a curriculum tweak.
- `stage_b_steps` controls the length of the attack-conditioning phase between the clean phase (steps 0..5000) and the ramp tail (steps 5000+stage_b_steps..30000). Lengthening it from 10000 to 14000 means attacks are still being introduced at moderate severity until step 19000 (vs step 15000 in the parent) before reaching α=1.0 stability, giving the masked L1 and JND-YUV terms more time to co-adapt with the bit pathway under noise. This is the regime where the parent's `bit_acc_clean_min=0.9949` weakness lives, and EMA's smoothing should be most useful here.
- Total steps remain locked at 30000 (`--max_steps`, `--total_steps`); only the internal split shifts. The clean phase length (`--stage_a_steps`) is held at 5000 so that EMA warmup behavior is *identical to design001*, preserving the ablation's diagnostic value.

Hypothesis: vs design001 of this idea, this design recovers `bit_acc_clean_min` further (more attack-conditioning time → more robust bit pathway → cleaner eval at the EMA weights) at small or zero PSNR cost. Vs the parent (no EMA), the longer `stage_b_steps` alone would slightly extend training noise; EMA absorbs that, so the EMA + lengthened curriculum is expected to dominate both.

## 2. CLI flags / config for `baseline/exp0_inn_train.py`

All flags identical to design001 of this idea **except** `--stage_b_steps`. **No locked flag changed.**

```
--num_blocks 16
--subnet_type rdb
--lr 2e-4
--img_weight 1.0
--bit_weight 20.0
--stage_a_steps 5000
--stage_b_steps 14000           # changed from 10000
--batch_size 16        # locked
--secret_len 896       # locked
--resolution 256       # locked
--max_steps 30000      # locked
--total_steps 30000    # locked
```

No new CLI flags. EMA decay (0.999) and EMA warmup (= `stage_a_steps = 5000`) are the same hard-coded constants as design001. `jnd_beta=4.0`, `pixel_w=0.5`, `jnd_mode="sobel"` unchanged.

## 3. Files Builder may modify

- `code/exp0_inn_train.py`

Builder must **not** touch any of:
- `code/inn_model.py`, `code/losses.py`, `code/train.py`
- `infra/**` (locked)

## 4. Algorithmic spec — exact changes to `code/exp0_inn_train.py`

Identical to design001 of this idea, sections §4.1 through §4.7 verbatim:

- §4.1 — `WeightEMA` class
- §4.2 — `EMA_DECAY = 0.999`, `EMA_WARMUP_STEPS = args.stage_a_steps`
- §4.3 — construct `ema = WeightEMA(model, decay=EMA_DECAY)` after model print
- §4.4 — dual-slot resume (online first, then EMA)
- §4.5 — `if global_step >= EMA_WARMUP_STEPS: ema.update(model)` after `sched.step()`
- §4.6 — `save_ckpt()` writes EMA into `ckpt["model"]`, online into `ckpt["model_online"]`
- §4.7 — `ema_active` field in `metrics.jsonl`

The only delta vs design001 of this idea is the CLI flag value `--stage_b_steps 14000`, which is consumed by the existing `NoiseCurriculum` (no code change to `NoiseCurriculum` itself; the curriculum already takes `stage_b_steps` as a parameter).

The `ImageSecretLoss(...)` constructor call is unchanged from the parent / design001 of this idea: `ImageSecretLoss(secret_weight=args.bit_weight, yuv_w=1.5, lpips_w=0.0, ffl_w=1.5, jnd_beta=4.0, pixel_w=0.5, jnd_mode="sobel").to(device)`.

## 5. Expected behavior after the change

- `metrics.jsonl` per-step keys = parent keys ∪ `{ema_active}`. The `alpha` curriculum field will visibly differ from design001 around steps 5000–19000 (longer attack-conditioning phase).
- `ckpt_latest.pt` payload keys = parent keys ∪ `{model_online, ema}`. `ckpt["model"]` holds EMA weights for `step >= 5000`.
- Eval `metrics.csv`: target `score >= 28.4`, with explicit improvement on `bit_acc_clean_min` over both the parent (0.9949) and design001 of this idea. Hard condition: `bit_acc_clean = 1.000` at every one of the 9 resolutions for the EMA weights at end of training.
- Wallclock: same as design001 (`stage_b_steps` only changes the curriculum α schedule, not the per-step compute cost). Well within the 12 h SLURM budget.

## 6. Constraints and invariants Builder must preserve

1. **Locked CLI flags unchanged:** `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.
2. **Locked infra unchanged:** zero edits under `infra/**`.
3. **Single perturbation axis:** `--stage_b_steps 14000` is the *only* knob changed vs design001 of this idea. Builder must NOT also change `--lr`, `--num_blocks`, `--subnet_type`, `--img_weight`, `--bit_weight`, `--stage_a_steps`, `pixel_w`, `jnd_beta`, or any loss weight. Stacking perturbations destroys the diagnostic value of the comparison.
4. **No edits to `code/losses.py` or `code/inn_model.py`** — JND+L1 mask reused verbatim from parent.
5. **No new CLI flags.** `EMA_DECAY` and `EMA_WARMUP_STEPS` are module-level constants. `--stage_b_steps` is an *existing* CLI flag (not new); only its value changes.
6. **Hard condition:** `bit_acc_clean = 1.000` at every eval resolution must hold for the EMA weights at end of training.
7. EMA semantics, dual-slot ckpt, resume order, and autocast separation per design001 of this idea §4.

## 7. Edge cases

- **Curriculum boundary check:** `stage_a_steps=5000` + `stage_b_steps=14000` = 19000 steps before the ramp tail begins; tail length = 30000 - 19000 = 11000 steps (vs the parent's 15000-step tail). Builder must verify `NoiseCurriculum`'s α schedule still reaches α=1.0 well before step 30000 (it does: the existing tail-ramp logic finishes within ~stage_b_steps/2 after the boundary). If the existing `NoiseCurriculum` has any hard-coded assumption that `stage_a_steps + stage_b_steps <= 15000`, flag it to Designer — do not silently fix.
- **EMA warmup invariance:** EMA still starts at step 5000, identical to design001 of this idea. The longer `stage_b_steps` means the EMA covers more of the attack-conditioning phase, which is exactly the regime where smoothing is hypothesised to help.
- **Wallclock parity:** `stage_b_steps` does not change per-step cost, only the α schedule consumed in the noise injection step. If wallclock differs from design001 by more than 5%, flag.
- **`bit_acc_clean_min` regression check:** as in design001 of this idea, if EMA weights at step 30000 give `bit_acc_clean_min < 0.9949` at any resolution, flag as a regression in the implementation summary.
