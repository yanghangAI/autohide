**Design Description:** EMA shadow weights only — maintain a fp32 EMA copy of `INNCodec` parameters with decay 0.999, start updating after the clean phase (`step >= stage_a_steps`), and write the EMA weights to the eval-consumed `model` slot in `ckpt_latest.pt`. No other change to loss, curriculum, INN, or optimizer.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Goal

Isolate the EMA-weight-averaging effect on the score axis. Expectation: +0.2 to +0.5 score over baseline (27.27) via a small PSNR bump (cleaner, less-noisy weights at eval time) with `bit_acc_clean = 1.000` preserved at every eval resolution.

## 2. CLI flags / config for `baseline/exp0_inn_train.py`

This section enumerates the full training-run config (all CLI / config-level knobs).

All flags identical to baseline canonical sbatch invocation. **No locked flag changed.**

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

No new CLI flags are added. EMA decay (0.999) and EMA-warmup-step (= `stage_a_steps`) are hard-coded constants inside the modified training script. Hard-coding (rather than a CLI knob) is intentional: the locked-flag set must not grow, and design002/design003 deliberately use the same constant so the three designs are directly comparable.

## 3. Files Builder may modify

Builder may only touch the following files inside `code/` (mirroring `baseline/`):

- `code/exp0_inn_train.py`

Builder must **not** touch any of:
- `code/inn_model.py`, `code/losses.py` (no model or loss change in this design)
- `infra/**` (locked)

## 4. Algorithmic spec — exact changes to `code/exp0_inn_train.py`

### 4.1 Add an EMA helper class (top of file, after the `_psnr` function, before `class NoiseCurriculum`)

```python
class WeightEMA:
    """Fp32 exponential-moving-average shadow of a model's parameters.

    - Holds one fp32 buffer per parameter on the same device as the param.
    - update() runs *outside* autocast in fp32 to avoid drift from low-
      precision averaging.
    - Buffers (BatchNorm running stats, etc.) are copied verbatim every
      step (no averaging) — INNCodec has no BN but this future-proofs.
    - state_dict() / load_state_dict() round-trip the shadow tensors plus
      the integer `num_updates` counter so resume is bit-exact.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters() if p.requires_grad
        }
        self.buffers = {
            name: b.detach().clone()
            for name, b in model.named_buffers()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)
        for name, b in model.named_buffers():
            self.buffers[name].copy_(b.detach())
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Overwrite `model` parameters/buffers with the EMA shadow."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(dtype=p.dtype))
        for name, b in model.named_buffers():
            if name in self.buffers:
                b.data.copy_(self.buffers[name])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
            "buffers": self.buffers,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd["decay"])
        self.num_updates = int(sd["num_updates"])
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}
        self.buffers = {k: v.clone() for k, v in sd["buffers"].items()}
```

### 4.2 Constants near top of `main()`

Immediately after `args = ap.parse_args()` (around line 233), add:

```python
EMA_DECAY = 0.999
EMA_WARMUP_STEPS = args.stage_a_steps  # start EMA only after the clean phase
```

### 4.3 Construct EMA after model construction (around line 250)

Right after the `print(f"[model] INNCodec n_params=...")` line, add:

```python
ema = WeightEMA(model, decay=EMA_DECAY)
print(f"[ema] enabled: decay={EMA_DECAY} warmup_steps={EMA_WARMUP_STEPS}")
```

### 4.4 Resume — load EMA state if present (inside the `if ckpt_path.exists():` block, after the existing `model.load_state_dict(ck["model"])` etc.)

Add:

```python
if "ema" in ck:
    ema.load_state_dict(ck["ema"])
    print(f"[resume] EMA restored: num_updates={ema.num_updates}")
else:
    # Old-style ckpt (no EMA); rebuild EMA from current online weights.
    ema = WeightEMA(model, decay=EMA_DECAY)
    print("[resume] no EMA in ckpt; reinitialised from current online weights")
```

**Important:** the resume payload's `ck["model"]` must hold the **online (non-EMA) weights** for resume correctness. See §4.6 for the dual-slot save format that achieves this.

### 4.5 Update EMA inside the training loop

Inside the `while global_step < args.max_steps:` loop, **after** `sched.step()` and **before** the `with torch.no_grad():` diagnostics block (i.e. immediately after line 385), add:

```python
if global_step >= EMA_WARMUP_STEPS:
    ema.update(model)
```

The `>=` means EMA begins updating on the first post-clean-phase step, exactly as the idea prescribes ("warm up EMA only after the clean phase completes").

### 4.6 `save_ckpt()` — write EMA weights to the eval-consumed `model` slot

Replace the body of `save_ckpt()` so that the `model` slot holds the EMA weights (which `infra/exp0_inn_eval.py` consumes), while the online weights are preserved separately for resume. Concretely, change the `payload` dict to:

```python
def save_ckpt():
    tmp = ckpt_path.with_suffix(".pt.tmp")
    # Eval reads ckpt["model"]; write the EMA weights into that slot.
    # If EMA has not warmed up yet (step < stage_a_steps) the EMA shadow
    # equals the online init+clean weights from construction, so writing
    # the online state is the safer fallback for that pre-warmup window.
    if global_step >= EMA_WARMUP_STEPS:
        eval_state = {k: v.detach().cpu() for k, v in ema.shadow.items()}
        # Merge in non-parameter buffers (model.state_dict has both).
        full_online = model.state_dict()
        for k in full_online:
            if k not in eval_state:
                eval_state[k] = full_online[k].detach().cpu()
    else:
        eval_state = model.state_dict()

    payload = {
        "model": eval_state,                 # consumed by eval (EMA after warmup)
        "model_online": model.state_dict(),  # for resume
        "ema": ema.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "acc_ema": acc_ema,
        "psnr_ema": psnr_ema,
        "l1_ema": l1_ema,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "py_rng": random.getstate(),
        "args": vars(args),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(payload, tmp)
    os.replace(tmp, ckpt_path)
```

### 4.7 Resume — restore the online weights into `model`

In the resume block (§4.4), the existing `model.load_state_dict(ck["model"])` would now load the *EMA* weights into the online model, which corrupts the optimizer state. Replace with:

```python
if "model_online" in ck:
    model.load_state_dict(ck["model_online"])
else:
    # First-resume from a baseline-style ckpt (no online slot).
    model.load_state_dict(ck["model"])
```

This must come *before* the EMA-restore call in §4.4.

## 5. Logging spec

Add one diagnostic field to the per-step `rec` dict (around line 413):

```python
ema_active=int(global_step >= EMA_WARMUP_STEPS),
```

No other logging change. `metrics.jsonl` schema stays a strict superset of baseline.

## 6. Expected behaviour after the change

- Training cost: `+1` element-wise multiply-add per parameter per step (≈1.5M params → <1 ms/step on a 2080Ti). Run still completes well within the 12 h walltime; verify by inspecting `wallclock_s` in `metrics.jsonl` (baseline final ≈ 4–5 h on a single GPU; this design must not exceed 1.1× of that).
- `metrics.jsonl` per-step keys = baseline keys ∪ `{ema_active}`.
- `ckpt_latest.pt` payload keys = baseline keys ∪ `{model_online, ema}`. The `model` key now holds EMA weights for `step >= stage_a_steps` and online weights before that.
- `infra/exp0_inn_eval.py` (locked) reads `ckpt["model"]` unchanged; it will see EMA weights at end-of-training without any infra-side modification.
- Eval `metrics.csv`: target `score >= 27.5` (i.e. >= baseline + ~0.25). Hard condition: `bit_acc_clean = 1.000` at every one of the 9 resolutions. If the EMA weights fail this hard condition while the online weights pass it, the design is a regression even if `score` looks higher.

## 7. Invariants Builder must preserve

1. **Locked flags unchanged:** `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.
2. **Locked infra unchanged:** zero edits under `infra/**`. The eval script's contract (`ckpt["model"]` is the model state_dict) is honoured by writing EMA weights into that exact slot.
3. **Hard condition:** `bit_acc_clean = 1.000` at every eval resolution must hold for the **EMA** weights at end of training. The post-clean-phase EMA warm-up (`step >= stage_a_steps`) is the mitigation: by step 5000 the online model has already learned the clean pathway, and the EMA inherits that.
4. **Resume correctness:** an interrupted-then-resumed run must produce a checkpoint payload byte-identical (modulo timestamps) to an uninterrupted run with the same seed; this is enforced by the dual-slot `model` / `model_online` design and by EMA `state_dict` round-tripping the shadow tensors and `num_updates` counter.
5. **No autocast contamination:** `ema.update()` is called outside the autocast block and reads `p.detach().float()` so the shadow is always fp32.
6. **No new CLI flags.** `EMA_DECAY` and `EMA_WARMUP_STEPS` are module-level constants inside the modified train script.

## 8. Edge cases

- **Pre-warmup save:** if a checkpoint is saved at `step < stage_a_steps` (e.g. preemption at step 4000 with `ckpt_every=2000`), §4.6 writes the *online* weights to `ckpt["model"]`. Eval at that point would not be on EMA weights, but eval is only run at end-of-training (step 30000), well past the warmup, so this is purely a resume-safety property.
- **EMA decay vs. step count:** with decay=0.999 and 25 000 post-warmup updates, the effective half-life is ~693 steps and the EMA has fully forgotten its initialisation by step ~10 000 — so the final EMA is dominated by the converged-region weights, which is the regime where averaging helps most.
- **fp16 / autocast:** the EMA shadow is fp32 and `ema.update()` is outside autocast; the model's online parameters may be fp32 (AdamW master weights are fp32 even with AMP), so `p.detach().float()` is a no-op cast in the common case but defensive in case of future fp16 master-weight changes.
