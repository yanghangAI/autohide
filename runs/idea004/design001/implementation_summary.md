**Files changed:**
code/exp0_inn_train.py

**Changes:**
- `code/exp0_inn_train.py`: Added a `WeightEMA` helper class (fp32 shadow buffer, decay 0.999) and integrated it into `main()` per design001 §4. The EMA is constructed right after the model, updated every step once `global_step >= stage_a_steps`, written to the eval-consumed `ckpt["model"]` slot of `save_ckpt()` (with `ckpt["model_online"]` preserved for resume), and a one-shot `ema_active` integer is added to per-step `metrics.jsonl` records. The resume block now restores online weights into `model` and rebuilds/loads the EMA shadow.

In `code/exp0_inn_train.py`, the EMA helper class:

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
```

In `code/exp0_inn_train.py`, EMA constants and construction:

```python
    EMA_DECAY = 0.999
    EMA_WARMUP_STEPS = args.stage_a_steps  # start EMA only after the clean phase
```

In `code/exp0_inn_train.py`, EMA construction:

```python
    ema = WeightEMA(model, decay=EMA_DECAY)
    print(f"[ema] enabled: decay={EMA_DECAY} warmup_steps={EMA_WARMUP_STEPS}")
```

In `code/exp0_inn_train.py`, the EMA update inside the training loop:

```python
        if global_step >= EMA_WARMUP_STEPS:
            ema.update(model)
```

In `code/exp0_inn_train.py`, the dual-slot `save_ckpt()`:

```python
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
```

In `code/exp0_inn_train.py`, the resume block:

```python
        if "model_online" in ck:
            model.load_state_dict(ck["model_online"])
        else:
            # First-resume from a baseline-style ckpt (no online slot).
            model.load_state_dict(ck["model"])
        if "ema" in ck:
            ema.load_state_dict(ck["ema"])
            print(f"[resume] EMA restored: num_updates={ema.num_updates}")
        else:
            # Old-style ckpt (no EMA); rebuild EMA from current online weights.
            ema = WeightEMA(model, decay=EMA_DECAY)
            print("[resume] no EMA in ckpt; reinitialised from current online weights")
```

In `code/exp0_inn_train.py`, the new logging field:

```python
                       ema_active=int(global_step >= EMA_WARMUP_STEPS),
```
