**Files changed:**
- `code/exp0_inn_train.py`

**Changes:**
- `code/exp0_inn_train.py` — added `import kornia.augmentation as K`; added module-level helper classes `_DetDownUpResize` and `_HardChain`; extended `NoiseCurriculum` init with the three-chain bank and `_p_hard_max`; rewrote `attack(step)` to branch into a `p_hard(α)`-gated deterministic hard chain or fall through to the existing random `AttackLayer`; updated `to(device)` to also push the hard-chain bank; added a startup hard-chain sanity probe; logged the per-step attack branch into `metrics.jsonl` and the per-step stdout line.

The new helper classes added to `code/exp0_inn_train.py`:

```python
class _DetDownUpResize(torch.nn.Module):
    """Deterministic down-then-up bilinear resize at a fixed scale.
    Local re-implementation of the logic in infra/attacks.py so we do not
    import from infra/exp0_inn_eval.py's private classes. Preserves spatial
    dimensions; fully differentiable."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        new_h = max(8, int(round(h * self.scale)))
        new_w = max(8, int(round(w * self.scale)))
        if (new_h, new_w) == (h, w):
            return x
        small = F.interpolate(x, size=(new_h, new_w), mode="bilinear",
                              align_corners=False)
        return F.interpolate(small, size=(h, w), mode="bilinear",
                             align_corners=False)


class _HardChain(torch.nn.Module):
    """Deterministic resize→JPEG chain at a fixed (scale, q) operating
    point. Mirrors infra/exp0_inn_eval.py::_ResizeJPEGChain semantics:
    input/output in [-1, 1], spatial dims preserved, fp16-safe via
    sanitize-and-cast at entry."""

    def __init__(self, scale: float, q: int):
        super().__init__()
        self.resize = _DetDownUpResize(scale)
        self.jpeg = K.RandomJPEG(jpeg_quality=(q, q), p=1.0)
        self.scale = float(scale)
        self.q = int(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        y = ((y + 1.0) * 0.5).clamp(0.0, 1.0)
        y = self.resize(y)
        y = self.jpeg(y).clamp(0.0, 1.0)
        return y * 2.0 - 1.0
```

The bank construction added inside the curriculum constructor in `code/exp0_inn_train.py`:

```python
        self._hard_chains = torch.nn.ModuleList([
            _HardChain(scale=0.5, q=85),  # eval chain_insta operating point
            _HardChain(scale=0.5, q=70),  # eval chain_whatsapp_std operating point
            _HardChain(scale=0.3, q=50),  # eval chain_wechat operating point
        ])
        self._p_hard_max = 0.4   # asymptotic hard-chain firing probability at α=1
        self.last_branch = "clean"  # diagnostic for logging
```

The new branched attack and updated to-device in `code/exp0_inn_train.py`:

```python
    def attack(self, step: int):
        a = self.alpha(step)
        if a == 0.0:
            self.last_branch = "clean"
            return None
        p_hard = self._p_hard_max * a
        if torch.rand(1).item() < p_hard:
            idx = int(torch.randint(0, len(self._hard_chains), (1,)).item())
            self.last_branch = f"hard[{self._hard_chains[idx].scale:.1f},q{self._hard_chains[idx].q}]"
            return self._hard_chains[idx]
        # Random branch: scale Bernoulli probabilities by α as before.
        self._attack.jpeg1.p = self._p_max["jpeg1"] * a
        self._attack.jpeg2.p = self._p_max["jpeg2"] * a
        self._attack.resize.p = self._p_max["resize"] * a
        self.last_branch = "random"
        return self._attack

    def to(self, device):
        self._attack.to(device)
        self._hard_chains.to(device)
        return self
```

The startup sanity probe added in main inside `code/exp0_inn_train.py`:

```python
    # Probe each hard chain once at fp16 with NaN/Inf-poisoned input so any
    # kornia/dtype bug surfaces at startup, not at step 5001.
    _probe = torch.zeros(2, 3, args.resolution, args.resolution, device=device, dtype=torch.float16)
    _probe[0, 0, 0, 0] = float("nan")
    _probe[0, 0, 0, 1] = float("inf")
    with torch.no_grad():
        for hc in curriculum._hard_chains:
            _ = hc(_probe)
    print("[init] hard-chain sanity check passed.")
```

The logging additions inside `code/exp0_inn_train.py`:

```python
                       attack_branch=curriculum.last_branch,
```

```python
                  f"img={img_loss.item():.4f} bit={bit_loss.item():.4f} "
                  f"br={curriculum.last_branch}")
```
