**Design Description:** Pure composition — take `runs/idea003/design003` end-to-end (Sobel JND mask on YUV term + masked L1 RGB pixel term, β=4.0, pixel_w=0.5, jnd_mode="sobel") and stack the `WeightEMA` shadow-weight machinery from `runs/idea004/design001` verbatim (decay=0.999, EMA warmup at `stage_a_steps=5000`, dual-slot `model`/`model_online` checkpoint). No other change to loss, curriculum, INN, or optimizer.

**Parent:** runs/idea003/design003

**Starting Point:** runs/idea003/design003

---

## 1. Goal

Isolate the additive-stacking effect of EMA on top of the current best `Done` configuration. Hypothesis: EMA's smoothing recovers small amounts of both PSNR and bit_acc (especially `bit_acc_clean_min`, which dipped to 0.9949 in the parent), giving a positive `score` delta over the parent's 28.16. Designs 002 and 003 use this as their reference point.

## 2. CLI flags / config for `baseline/exp0_inn_train.py`

All flags identical to the parent (`runs/idea003/design003`). **No locked flag changed.**

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

No new CLI flags. EMA decay (0.999) and EMA warmup step (= `stage_a_steps`) are hard-coded constants inside the modified train script, exactly as in `idea004/design001`. `jnd_beta=4.0`, `pixel_w=0.5`, `jnd_mode="sobel"` remain hard-coded inside `ImageSecretLoss(...)` exactly as in the parent.

## 3. Files Builder may modify

Builder may only touch the following files inside `code/` (mirroring `baseline/`):

- `code/exp0_inn_train.py`

Builder must **not** touch any of:
- `code/inn_model.py` (no architecture change in this design)
- `code/losses.py` (no loss change in this design — the parent already provides the JND+L1 mask)
- `code/train.py`
- `infra/**` (locked)

The `code/losses.py` and `code/inn_model.py` inherited from the parent `runs/idea003/design003` are taken verbatim and must not be edited.

## 4. Algorithmic spec — exact changes to `code/exp0_inn_train.py`

The Builder copies the parent's `code/exp0_inn_train.py` and applies the EMA edits from `runs/idea004/design001` §4 verbatim. Concretely:

### 4.1 Add `WeightEMA` class

Add the `WeightEMA` class exactly as in `runs/idea004/design001` §4.1, placed after the `_psnr` helper and before `class NoiseCurriculum`. Reproduced for unambiguity:

```python
class WeightEMA:
    """Fp32 exponential-moving-average shadow of a model's parameters."""

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

Immediately after `args = ap.parse_args()`, add:

```python
EMA_DECAY = 0.999
EMA_WARMUP_STEPS = args.stage_a_steps   # 5000 in this design
```

### 4.3 Construct EMA right after model construction

Right after the `print(f"[model] INNCodec n_params=...")` line, add:

```python
ema = WeightEMA(model, decay=EMA_DECAY)
print(f"[ema] enabled: decay={EMA_DECAY} warmup_steps={EMA_WARMUP_STEPS}")
```

### 4.4 Resume — restore online weights then EMA shadow

Inside the `if ckpt_path.exists():` block, replace `model.load_state_dict(ck["model"])` with the dual-slot logic, executed in this exact order (online first, then EMA):

```python
if "model_online" in ck:
    model.load_state_dict(ck["model_online"])
else:
    # First-resume from a baseline-style or parent-style ckpt (no online slot).
    model.load_state_dict(ck["model"])

if "ema" in ck:
    ema.load_state_dict(ck["ema"])
    print(f"[resume] EMA restored: num_updates={ema.num_updates}")
else:
    ema = WeightEMA(model, decay=EMA_DECAY)
    print("[resume] no EMA in ckpt; reinitialised from current online weights")
```

### 4.5 Update EMA inside the training loop

Inside `while global_step < args.max_steps:`, after `sched.step()` and before the per-step `with torch.no_grad():` diagnostics block, add:

```python
if global_step >= EMA_WARMUP_STEPS:
    ema.update(model)
```

### 4.6 `save_ckpt()` — write EMA weights into the eval-consumed `model` slot

Replace `save_ckpt()` body with the dual-slot payload from `runs/idea004/design001` §4.6:

```python
def save_ckpt():
    tmp = ckpt_path.with_suffix(".pt.tmp")
    if global_step >= EMA_WARMUP_STEPS:
        eval_state = {k: v.detach().cpu() for k, v in ema.shadow.items()}
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

### 4.7 Per-step diagnostic field

Add one field to the per-step `rec` dict written to `metrics.jsonl`:

```python
ema_active=int(global_step >= EMA_WARMUP_STEPS),
```

No other logging change. The parent's `l_pixel`, `l_yuv_unweighted`, `jnd_beta`, `pixel_w`, `jnd_mode` fields are kept verbatim.

## 5. Expected behavior after the change

- Training cost: ~+1 ms/step on ~1.5M params; well under the 12 h SLURM budget.
- `metrics.jsonl` per-step keys = parent keys ∪ `{ema_active}`.
- `ckpt_latest.pt` payload keys = parent keys ∪ `{model_online, ema}`. `ckpt["model"]` holds EMA weights for `step >= 5000` and online weights before that.
- `infra/exp0_inn_eval.py` (locked) reads `ckpt["model"]` unchanged and will see the EMA weights at end of training.
- Eval `metrics.csv`: target `score >= 28.4` (parent 28.16 + ~0.25 from EMA smoothing). Hard condition: `bit_acc_clean = 1.000` at every one of the 9 eval resolutions for the EMA weights at end of training.
- Soft expectation: `bit_acc_clean_min` recovers from the parent's 0.9949 toward 1.0, since EMA smooths the late-training oscillations introduced by the masked L1 term.

## 6. Constraints and invariants Builder must preserve

1. **Locked CLI flags unchanged:** `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.
2. **Locked infra unchanged:** zero edits under `infra/**`. The eval script's contract (`ckpt["model"]` is the model state_dict consumed at eval) is honoured by writing EMA weights into that exact slot.
3. **No edits to `code/losses.py` or `code/inn_model.py`** — the parent's JND+L1-masked loss is reused verbatim. Builder must not redefine `ImageSecretLoss`, must not change `jnd_beta`, `pixel_w`, `jnd_mode`, `yuv_w`, `lpips_w`, `ffl_w`, or `secret_weight=20.0`.
4. **No new CLI flags.** `EMA_DECAY` and `EMA_WARMUP_STEPS` are module-level constants inside the modified train script.
5. **Hard condition:** `bit_acc_clean = 1.000` at every eval resolution must hold for the **EMA** weights at end of training. The post-clean-phase EMA warmup (`step >= stage_a_steps=5000`) is the structural mitigation: by step 5000 the online model has already learned the clean pathway and EMA inherits it.
6. **Resume correctness:** §4.4's dual-slot resume must execute online-then-EMA in that exact order. An interrupted-then-resumed run must produce a checkpoint payload byte-identical (modulo timestamps) to an uninterrupted run with the same seed.
7. **No autocast contamination:** `ema.update()` is called outside the autocast block and reads `p.detach().float()` so the shadow is always fp32.
8. The `ImageSecretLoss(...)` constructor call in `code/exp0_inn_train.py` is unchanged from the parent: `ImageSecretLoss(secret_weight=args.bit_weight, yuv_w=1.5, lpips_w=0.0, ffl_w=1.5, jnd_beta=4.0, pixel_w=0.5, jnd_mode="sobel").to(device)`.

## 7. Edge cases

- **Pre-warmup save:** if a checkpoint is saved at `step < 5000` (e.g. preemption at step 4000 with `ckpt_every=2000`), §4.6 writes the *online* weights to `ckpt["model"]`. End-of-training eval is at step 30000, well past warmup, so this is purely a resume-safety property.
- **EMA decay vs. step count:** with decay=0.999 and 25 000 post-warmup updates, the effective half-life is ~693 steps; the final EMA is dominated by the converged-region weights.
- **fp16 / autocast:** EMA shadow is fp32 and `ema.update()` is outside autocast. The masked L1 / YUV terms in the parent loss already operate inside the parent's existing fp32-wrapped scope; this design changes none of that.
- **Mask reuse with broadcast** (inherited from parent): `w` shape is `(B, 1, H, W)`; broadcast over RGB. Builder must not alter the parent's mask code.
- **`bit_acc_clean_min` regression:** if EMA weights at step 30000 give `bit_acc_clean_min < parent's 0.9949` at any resolution, flag in implementation summary as a regression even if `score` is higher; this is diagnostic for design002 and design003.
