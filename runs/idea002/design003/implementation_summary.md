**Files changed:**
- `code/exp0_inn_train.py`

**Changes:**
- `code/exp0_inn_train.py` — added `import kornia.augmentation as K`; added module-level helper classes `_DetDownUpResize` and `_HardChain` (verbatim from design001 spec); extended `NoiseCurriculum` init with the 5-entry fixed bank, uniform initial weights, and per-chain bit-acc cache; rewrote `attack(step)` to sample bank entries by current weights; added `update_weights()` (no-grad, scores each chain on a held-out probe and re-weights via softmax of `(1 - bit_acc)/τ` with floor 0.05); updated `to(device)` to move bank chains and weight tensors; added a startup bank sanity probe; added probe-batch construction in `main()` from `val_paths`; added periodic mining call every 500 steps gated on `step >= ramp_start`; added `bank_weights` to checkpoint payload and resume restore; logged `attack_branch`, `bank_idx`, `bank_weights` into `metrics.jsonl` and `br=<branch>` into the per-step stdout line.

The new helper classes added to `code/exp0_inn_train.py`:

```python
class _DetDownUpResize(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)
    def forward(self, x):
        h, w = x.shape[-2:]
        new_h = max(8, int(round(h * self.scale)))
        new_w = max(8, int(round(w * self.scale)))
        if (new_h, new_w) == (h, w):
            return x
        small = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


class _HardChain(torch.nn.Module):
    def __init__(self, scale: float, q: int):
        super().__init__()
        self.resize = _DetDownUpResize(scale)
        self.jpeg = K.RandomJPEG(jpeg_quality=(q, q), p=1.0)
        self.scale = float(scale)
        self.q = int(q)
        self.name = f"hard_s{scale}_q{q}"
    def forward(self, x):
        y = x.float()
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        y = ((y + 1.0) * 0.5).clamp(0.0, 1.0)
        y = self.resize(y)
        y = self.jpeg(y).clamp(0.0, 1.0)
        return y * 2.0 - 1.0
```

The bank construction added inside the curriculum constructor in `code/exp0_inn_train.py`:

```python
        # Fixed bank: index 0 is the existing random AttackLayer (called via a
        # small wrapper that returns its forward); indices 1..4 are deterministic
        # eval-mirrored chains. Builder MUST keep this order so logging is stable.
        self._bank = [
            ("random", self._attack),
            ("hard_s0.5_q85", _HardChain(scale=0.5, q=85)),
            ("hard_s0.5_q70", _HardChain(scale=0.5, q=70)),
            ("hard_s0.4_q75", _HardChain(scale=0.4, q=75)),  # mirrors chain_x
            ("hard_s0.3_q50", _HardChain(scale=0.3, q=50)),
        ]
        # Initial weights: uniform over the 5 entries. Updated by mining.
        self._bank_weights = torch.full((len(self._bank),), 1.0 / len(self._bank))
        self._bank_bit_acc = torch.full((len(self._bank),), 0.5)  # for logging
        self.last_bank_idx = 0
        self.last_branch = "clean"
```

The new bank-sampling `attack(step)`, `update_weights`, and updated `to(device)` in `code/exp0_inn_train.py`:

```python
    def attack(self, step: int):
        a = self.alpha(step)
        if a == 0.0:
            self.last_branch = "clean"
            self.last_bank_idx = 0
            return None
        # Sample one bank entry by current weights.
        idx = int(torch.multinomial(self._bank_weights, num_samples=1).item())
        self.last_bank_idx = idx
        name, mod = self._bank[idx]
        if name == "random":
            # Mutate Bernoulli probabilities by α₁ exactly as today.
            self._attack.jpeg1.p = self._p_max["jpeg1"] * a
            self._attack.jpeg2.p = self._p_max["jpeg2"] * a
            self._attack.resize.p = self._p_max["resize"] * a
            self.last_branch = "random"
            return self._attack
        self.last_branch = name
        return mod

    @torch.no_grad()
    def update_weights(self, model, probe_cover: torch.Tensor,
                       probe_secret: torch.Tensor, alpha: float,
                       tau: float = 0.05, floor: float = 0.05) -> None:
        """Score each bank entry on the probe and re-weight by softmax of
        (1 - bit_acc) / tau, with a minimum-weight floor so no chain is
        starved entirely. Caller is responsible for switching the model to
        eval mode and back."""
        device = probe_cover.device
        container = model.embed(probe_cover, probe_secret).clamp(-1, 1)
        container = torch.nan_to_num(container, nan=0.0, posinf=1.0, neginf=-1.0)
        accs = []
        for i, (name, mod) in enumerate(self._bank):
            if name == "random":
                # Probe at the current α (worst expected severity).
                self._attack.jpeg1.p = self._p_max["jpeg1"] * alpha
                self._attack.jpeg2.p = self._p_max["jpeg2"] * alpha
                self._attack.resize.p = self._p_max["resize"] * alpha
            attacked = mod(container)
            logits = model.extract(attacked)
            acc = ((logits > 0).float() == probe_secret).float().mean()
            accs.append(acc)
        accs_t = torch.stack(accs).to(device)
        self._bank_bit_acc = accs_t.detach().cpu()
        # Softmax over (1 - acc) / tau, then floor + renormalize.
        logits_w = (1.0 - accs_t) / tau
        w = torch.softmax(logits_w, dim=0).cpu()
        w = w.clamp_min(floor)
        w = w / w.sum()
        self._bank_weights = w

    def to(self, device):
        self._attack.to(device)
        for _, m in self._bank[1:]:  # skip the random entry, already moved above
            m.to(device)
        self._bank_weights = self._bank_weights.to(device)
        self._bank_bit_acc = self._bank_bit_acc.to(device)
        return self
```

The startup bank sanity probe added in main inside `code/exp0_inn_train.py`:

```python
    _probe = torch.zeros(2, 3, args.resolution, args.resolution, device=device, dtype=torch.float16)
    _probe[0, 0, 0, 0] = float("nan")
    _probe[0, 0, 0, 1] = float("inf")
    with torch.no_grad():
        for name, mod in curriculum._bank[1:]:
            _ = mod(_probe)
    print(f"[init] bank sanity check passed: {[n for n, _ in curriculum._bank]}")
```

The probe-batch construction added in main inside `code/exp0_inn_train.py`:

```python
    # Mining probe: a single fixed batch from the held-out val set.
    # Same source as inline validation so no new data path is introduced.
    mining_probe = None
    if val_paths:
        n_probe = min(args.batch_size, len(val_paths))
        probe_cover = _val_load_to_size(val_paths[:n_probe], 256, 256).to(device)
        probe_secret = torch.randint(
            0, 2, (n_probe, args.secret_len), device=device, dtype=torch.float32,
        )
        mining_probe = (probe_cover, probe_secret)
        print(f"[mining] probe batch ready: {n_probe} images @ 256x256")
    else:
        print("[mining] disabled: no val images; falling back to uniform bank weights")
```

The resume-block restore added inside `code/exp0_inn_train.py`:

```python
        # Restore mining bank weights if present; default to uniform.
        bw = ck.get("bank_weights", None)
        if bw is not None:
            try:
                bw_t = torch.tensor(list(bw), dtype=torch.float32)
                if bw_t.numel() == len(curriculum._bank):
                    curriculum._bank_weights = bw_t.to(curriculum._bank_weights.device)
            except Exception as e:
                print(f"[resume] bank_weights restore skipped: {e}")
```

The checkpoint-payload addition inside `code/exp0_inn_train.py`:

```python
            "bank_weights": [float(w) for w in curriculum._bank_weights.tolist()],
```

The periodic mining call added inside `code/exp0_inn_train.py`:

```python
        MINING_EVERY = 500
        if mining_probe is not None and global_step >= curriculum.ramp_start \
                and global_step % MINING_EVERY == 0:
            was_training = model.training
            model.eval()
            curriculum.update_weights(
                model, mining_probe[0], mining_probe[1],
                alpha=curriculum.alpha(global_step),
            )
            if was_training:
                model.train()
            print(f"[mining {global_step}] weights="
                  f"{[f'{w:.2f}' for w in curriculum._bank_weights.tolist()]} "
                  f"bit_acc={[f'{a:.3f}' for a in curriculum._bank_bit_acc.tolist()]}")
```

The metrics-record additions and stdout addition in `code/exp0_inn_train.py`:

```python
                       attack_branch=curriculum.last_branch,
                       bank_idx=curriculum.last_bank_idx,
                       bank_weights=curriculum._bank_weights.tolist(),
```

```python
                  f"img={img_loss.item():.4f} bit={bit_loss.item():.4f} "
                  f"br={curriculum.last_branch}")
```
