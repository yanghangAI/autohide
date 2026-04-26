**Design Description:** EMA shadow weights (decay 0.999, warmup at `stage_a_steps`) **and** straight-through container clipping combined. Both interventions from design001 and design002 applied together. No other change to loss, curriculum, INN, or optimizer.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Goal

Quantify whether the EMA effect (design001) and the STE-clip effect (design002) are independent and additive, or whether they interact (positively or negatively). If additive, this should be the strongest configuration on the idea004 axis and the natural candidate to stack on the iter-2/iter-3 winner from the loss/curriculum axes. If negative interaction, this design's score (relative to design001 and design002) tells iter-3 which of the two single-intervention winners to stack and which to drop.

Parented on `baseline/` (not on design001 or design002) by design — per `idea.md`, "Built directly on `baseline/` — not on idea001/design003 — so that the gain from this axis is measured against the same starting point as idea001 and idea002 and is therefore additively comparable later."

## 2. CLI flags for `baseline/exp0_inn_train.py`

Identical to baseline canonical sbatch invocation. **No locked flag changed. No new CLI flags.**

```
--num_blocks 8
--subnet_type light
--lr 4e-4
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

EMA decay (0.999) and EMA warmup (= `stage_a_steps`) are hard-coded module-level constants, *identical* to design001. This identity is intentional so that the per-axis gains are directly comparable.

## 3. Files Builder may modify

- `code/exp0_inn_train.py`

Builder must **not** touch any of:
- `code/inn_model.py`, `code/losses.py` (no model or loss change)
- `infra/**` (locked)

## 4. Algorithmic spec — exact changes to `code/exp0_inn_train.py`

This design is the union of design001 §4 and design002 §4, applied to the same baseline starting point. The two interventions touch disjoint parts of the training script (EMA touches the optimizer/checkpoint side; STE touches the forward graph) and do not conflict.

### 4.1 EMA helper (identical to design001 §4.1)

Add the `WeightEMA` class verbatim from design001 §4.1 (top of file, after `_psnr`, before `class NoiseCurriculum`).

### 4.2 STE helper (identical to design002 §4.1)

Add the `ste_clamp` function verbatim from design002 §4.1 (top of file, immediately after `WeightEMA`, before `class NoiseCurriculum`).

### 4.3 Constants in `main()` (identical to design001 §4.2)

After `args = ap.parse_args()` (around line 233), add:

```python
EMA_DECAY = 0.999
EMA_WARMUP_STEPS = args.stage_a_steps
```

### 4.4 EMA construction (identical to design001 §4.3)

Right after the `print(f"[model] INNCodec n_params=...")` line, add:

```python
ema = WeightEMA(model, decay=EMA_DECAY)
print(f"[ema] enabled: decay={EMA_DECAY} warmup_steps={EMA_WARMUP_STEPS}")
```

### 4.5 Resume — load EMA + restore online weights into `model` (identical to design001 §4.4 and §4.7)

In the `if ckpt_path.exists():` block, replace `model.load_state_dict(ck["model"])` with:

```python
if "model_online" in ck:
    model.load_state_dict(ck["model_online"])
else:
    model.load_state_dict(ck["model"])

if "ema" in ck:
    ema.load_state_dict(ck["ema"])
    print(f"[resume] EMA restored: num_updates={ema.num_updates}")
else:
    ema = WeightEMA(model, decay=EMA_DECAY)
    print("[resume] no EMA in ckpt; reinitialised from current online weights")
```

### 4.6 Forward path — STE clip (identical to design002 §4.2)

Inside the `while global_step < args.max_steps:` loop, change line 371 from:

```python
container = container.clamp(-1.0, 1.0)
```

to:

```python
# STE clip + EMA shadow weights (idea004/design003 — combined).
container = ste_clamp(container, -1.0, 1.0)
```

The `_run_validation` function's `.clamp(-1, 1)` (line 158) stays as a hard clamp (per design002 §4.3 rationale).

### 4.7 EMA update in the loop (identical to design001 §4.5)

After `sched.step()` and before the `with torch.no_grad():` diagnostics block (around line 386), add:

```python
if global_step >= EMA_WARMUP_STEPS:
    ema.update(model)
```

### 4.8 `save_ckpt()` — dual-slot payload (identical to design001 §4.6)

Replace `save_ckpt()`'s body with the exact dual-slot version from design001 §4.6, so that `ckpt["model"]` holds EMA weights (post-warmup) and `ckpt["model_online"]` holds the online weights for resume.

## 5. Logging spec

Add **both** diagnostic fields to the per-step `rec` dict (union of design001 §5 and design002 §5):

```python
ema_active=int(global_step >= EMA_WARMUP_STEPS),
sat_frac=float((container_f.abs() >= 1.0 - 1e-6).float().mean().item()),
```

`metrics.jsonl` per-step schema = baseline keys ∪ `{ema_active, sat_frac}`.

## 6. Expected behaviour after the change

- Training cost: ≈ design001 cost + design002 cost = ≈ design001 cost (STE is free). Total walltime within 1.1× of baseline.
- `ckpt_latest.pt` payload keys = baseline keys ∪ `{model_online, ema}`.
- Eval `metrics.csv`: target `score >= 27.7` (i.e. >= baseline + ~0.4, on the assumption of approximate additivity of design001 and design002 expected gains). If the realised score is materially below `max(score(design001), score(design002))`, that quantifies negative interaction and is itself a useful finding for iter-3.
- Hard condition: `bit_acc_clean = 1.000` at every one of the 9 eval resolutions, on the **EMA** weights at end of training. STE preserves forward values, so it cannot by itself break the hard condition; EMA preserves it provided the warm-up condition is honoured.

## 7. Invariants Builder must preserve

1. **Locked flags unchanged:** `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.
2. **Locked infra unchanged:** zero edits under `infra/**`. EMA writes its weights into the eval-consumed `ckpt["model"]` slot exactly as in design001.
3. **Hard condition:** `bit_acc_clean = 1.000` at every eval resolution, on the EMA weights at end of training.
4. **Resume correctness:** dual-slot checkpoint format; an interrupted-then-resumed run produces the same end-of-training EMA and online weights (modulo timestamps) as an uninterrupted run with the same seed.
5. **No autocast contamination of EMA:** `ema.update()` is called outside autocast and reads `p.detach().float()`.
6. **STE only on train-time forward path** (one call site, line 371). Validation-path and eval-time clamps remain hard.
7. **No new CLI flags. No checkpoint key removed.**
8. **Identity with single-intervention designs:** the EMA helper, STE helper, EMA decay, EMA warmup step, and STE call site here must be byte-identical to those in design001 and design002 respectively, so that any difference in eval score between this design and the singletons is attributable purely to the interaction between the two interventions.

## 8. Edge cases

- **Pre-warmup save (`step < stage_a_steps`):** `ckpt["model"]` holds online weights (per design001 §4.6 fallback). End-of-training save is well past warmup so eval consumes EMA weights.
- **STE × EMA interaction at the boundary:** STE changes the trajectory through weight space (online weights take a different path than baseline); EMA averages along that trajectory. There is no a-priori reason these should interact pathologically — the EMA shadow is a low-pass filter of the online trajectory and is agnostic to *how* that trajectory is generated. If the runaway-`sat_frac` failure mode from design002 occurs here, the EMA shadow will lag behind the online drift and may actually be better than the online weights at eval time (a positive interaction); Builder should not preemptively cap or clip; just report.
- **Walltime sanity-check:** same as design001 (within 1.1× of baseline).
- **Comparison to design001/design002:** the three designs share parent (`baseline/`), seed default (`--seed 0`), and all CLI flags; the only differences are the code-side interventions described in §4 of each. This is what makes the per-axis decomposition meaningful.
