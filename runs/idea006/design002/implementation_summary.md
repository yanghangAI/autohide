**Files changed:**
code/exp0_inn_train.py

**Changes:**
- `code/exp0_inn_train.py`: Added the `WeightEMA` shadow-weight class (verbatim from `runs/idea004/design001`); added `EMA_DECAY=0.999` and `EMA_WARMUP_STEPS=args.stage_a_steps` constants in `main()`; constructed `ema = WeightEMA(model, decay=EMA_DECAY)` after model construction; updated the resume block to load `model_online`/`ema` slots with fallback; added `ema.update(model)` after `sched.step()` gated by warmup; rewrote `save_ckpt()` to write the EMA shadow into `ckpt["model"]` (eval slot) post-warmup and the live online weights into `ckpt["model_online"]`; added `ema_active` field to per-step `metrics.jsonl` records. No other change vs the parent (`runs/idea003/design003`); the JND+L1 loss config is unchanged.

In `code/exp0_inn_train.py`, the `WeightEMA` class:
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
```

In `code/exp0_inn_train.py`, the EMA constants and construction:
```python
    EMA_DECAY = 0.999
    EMA_WARMUP_STEPS = args.stage_a_steps   # 5000 in this design
```

```python
    ema = WeightEMA(model, decay=EMA_DECAY)
    print(f"[ema] enabled: decay={EMA_DECAY} warmup_steps={EMA_WARMUP_STEPS}")
```

In `code/exp0_inn_train.py`, the dual-slot resume:
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

In `code/exp0_inn_train.py`, the per-step EMA update:
```python
        if global_step >= EMA_WARMUP_STEPS:
            ema.update(model)
```

In `code/exp0_inn_train.py`, the dual-slot `save_ckpt()`:
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
```

In `code/exp0_inn_train.py`, the per-step `ema_active` log field:
```python
                       ema_active=int(global_step >= EMA_WARMUP_STEPS),
```
