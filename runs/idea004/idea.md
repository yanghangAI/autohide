**Idea Name:** Optimization-dynamics improvements: EMA shadow weights + straight-through container clipping

**Approach:** Improve the training-time optimization dynamics of the existing baseline pipeline — without touching the loss formulation, the curriculum, or the INN architecture — by (a) maintaining an exponential-moving-average copy of the model weights for inference-time use and (b) replacing the hard `container.clamp(-1, 1)` with a straight-through estimator so gradients flow through saturated pixels.

**Expected Designs:** 3

**Suggested Parent:** baseline/

**Baseline Source:** baseline/ (only `Done` clean-baseline entry; score=27.27, PSNR=27.33, bit_acc=0.9988, bit_acc_clean=1.0). Built directly on `baseline/` — not on idea001/design003 — so that the gain from this axis is measured against the same starting point as idea001 and idea002 and is therefore additively comparable later.

**Relationship to prior work:** New axis. Orthogonal to both prior ideas and to the extant idea003 (spatial-JND masking, which extends idea001/design003):
- idea001 attacked PSNR via the **loss formulation** (perceptual reweighting). Result: aggressive LPIPS collapsed training (design001=2.72, design002=16.36); YUV+FFL-only gave +0.46 (design003=27.73). Lesson: loss reshaping must be gentle.
- idea002 attacked bit_acc via the **attack curriculum** (eval-mirrored hard chains, severity-annealing, HEM bank). Result: mild changes give small wins (design001=27.44, design002=27.47); aggressive bank reweighting collapsed training entirely (design003 score≈0). Lesson: curriculum changes must be gentle.
- idea003 attacks PSNR via **spatial / content-adaptive loss reweighting** (a third loss-side axis that builds on idea001/design003).
- This idea changes neither the loss nor the curriculum nor the INN nor the cover-side weighting. It changes only **how the same loss + same curriculum + same model are optimized and how the trained weights are exposed to eval**. It therefore cannot conflict with any prior idea and can in principle stack on top of any of them in a later iteration.

**Motivation from the data and the codebase:**

1. **EMA shadow weights are a near-universal free win in image-quality models.** The training loop in `baseline/exp0_inn_train.py` uses AdamW + cosine schedule + grad-clip but has no weight averaging. The eval (`infra/exp0_inn_eval.py`, locked) consumes whatever weights are saved in the checkpoint. By maintaining `θ_ema ← decay·θ_ema + (1-decay)·θ` every step (typical decay 0.999–0.9999) and writing the EMA weights into the eval-consumed checkpoint slot, image-restoration / generative-model literature consistently reports 0.1–0.5 dB PSNR improvements with no training-time cost. At score sensitivity ≈ +1.0 score per +1 dB PSNR (since `score = 2·PSNR·(bit_acc-0.5)` and bit_acc ≈ 1), even +0.2 dB is worth ~+0.4 score — enough to surpass the current best (idea001/design003 = 27.73) on its own.

2. **Straight-through container clipping fixes a known gradient leak in the current loop.** In `baseline/exp0_inn_train.py` (around lines 370–372):
   ```python
   container = model.embed(cover, bits)
   container = container.clamp(-1.0, 1.0)
   attacked = attack_module(container) if attack_module is not None else container
   ```
   `clamp` is non-differentiable on the saturated set: whenever the INN pushes a pixel outside [-1, 1] (which happens routinely on high-contrast cover regions when the embedded residual is large), the gradient at that pixel is zeroed. The encoder cannot learn from the bit-loss signal on those exact pixels — precisely the pixels where it is over-embedding. A straight-through estimator (`container = container + (container.clamp(-1,1) - container).detach()`) preserves the clipped forward value while passing the upstream gradient unchanged in the backward pass. One-line change with well-understood semantics; should give the encoder a smoother optimization landscape near the [-1, 1] boundary, manifesting as either modestly higher PSNR (less over-embedding) or modestly higher bit_acc (encoder learns safer embedding magnitudes).

3. **Composability matters for the iter-2/iter-3 budget.** Both idea001 and idea002 each yielded one mildly-positive design on the table. Whichever ends up the leader after iter-2, an orthogonal optimization-dynamics idea can be stacked on it in iter-3 without re-tuning the loss or curriculum. That makes this a high-expected-value axis even if the per-design gain here is only +0.2 to +0.5 score.

4. **Failure mode is bounded.** Both EMA weights and STE clipping are conservative interventions: the worst-case behaviour of EMA is ≈ identical to the non-EMA model (decay shrinks toward online weights), and the worst-case behaviour of STE-clipping is ≈ identical to hard-clipping (forward values are byte-identical; only the gradient differs). The HEM-style collapse from idea002/design003 is not a risk profile this axis can reproduce.

**Suggested design directions (Designer to refine):**

1. **EMA-only.** Maintain `model_ema` with a single fixed decay (e.g. 0.999, ≈1000-step half-life — well-matched to the 30k-step run and the ~5k-step clean phase). Update every step; save EMA weights as the canonical eval-consumed `ckpt_latest.pt`. Keep online weights for the optimizer (in a separate slot in the checkpoint payload so resume still works). Verify `bit_acc_clean = 1.000` holds on the EMA weights at every eval resolution (the hard condition).

2. **STE-clip-only.** Replace the hard `container.clamp(-1.0, 1.0)` with a straight-through estimator that preserves the forward clip but passes the gradient through. No EMA. Isolates the gradient-flow effect from the weight-averaging effect.

3. **EMA + STE-clip combined.** Both interventions applied. If the two effects are independent and additive, this should be the strongest configuration. If they interact negatively, this design quantifies the interaction and tells iter-3 which to stack on the loss/curriculum winner.

**Constraints:**
- Parent is `baseline/`. Designs may modify `baseline/exp0_inn_train.py` (training loop: add EMA bookkeeping, add an STE-clip helper, change the checkpoint payload to include both online and EMA weights and write EMA weights to the eval-consumed `model` slot). A small EMA helper module under `baseline/` is acceptable if cleaner.
- Locked hyperparameters unchanged: `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`. EMA decay is a new design-level knob, not a locked hyperparameter — Designer may pick it (recommended search: {0.999, 0.9995, 0.9999}).
- Hard condition `bit_acc_clean = 1.000` at every eval resolution must hold. Risk-mitigation: warm up EMA only after the clean phase completes (`step >= stage_a_steps`) so the EMA weights start from a model that has already learned the clean pathway, rather than from the random-init noise floor.
- Loss formulation stays at the baseline default (pure-L1 host + BCE bits, `img_weight=1.0`, `bit_weight=20.0`). Curriculum stays at the baseline default (`stage_a_steps=5000`, `stage_b_steps=10000`, `severity="high"`). This isolates the optimization-dynamics axis from idea001's loss axis, idea002's curriculum axis, and idea003's spatial-mask axis so all four are independently measurable and later stackable.
- `--num_blocks` and `--subnet_type` stay at baseline defaults (8 DKiS / LightConvBlock per the canonical sbatch invocation). No INN-architecture changes — that is a separate axis reserved for a future idea.
- Optimizer / AMP / grad-clip stay at baseline defaults (`AdamW(lr=4e-4)`, `CosineAnnealingLR`, `clip_grad_norm_=5.0`, `amp=True`). EMA bookkeeping should be done in fp32 on a shadow buffer (not under autocast) to avoid drift from low-precision averaging — store the shadow on GPU to avoid host-device copies per step.
- Walltime: EMA adds one element-wise op per parameter per step on a model of ≈1.5M params — well below 1 ms/step. STE-clip is zero cost. Both fit comfortably within the 12 h walltime; Designer should still confirm by inspecting the `wallclock_s` field of the existing baseline `metrics.jsonl`.
