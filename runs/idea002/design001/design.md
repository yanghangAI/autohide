**Design Description:** Add an eval-mirrored "hard chain" branch to `NoiseCurriculum` that, with probability `p_hard(α)` ramped alongside the existing α-ramp, replaces the random `AttackLayer` sample with a deterministic resize→JPEG chain drawn uniformly from `{(0.5, 85), (0.5, 70), (0.3, 50)}`, giving the encoder direct gradient on the eval worst-case operating points without touching `infra/`.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Motivation and approach

`idea.md` shows the train/eval distribution gap is the dominant residual on `bit_acc`: the baseline `AttackLayer(severity="high")` co-fires (resize≤0.6, jpeg1 q≤50, jpeg2 firing) on roughly 1% of batches, so the encoder sees the WeChat-equivalent regime once per ~100 steps. The eval suite, in contrast, hits `chain_wechat=(scale=0.3, q=50)` and `chain_whatsapp_std=(scale=0.5, q=70)` on every test image, deterministically. This design closes the gap by adding a second branch into `NoiseCurriculum.attack(step)`: with probability `p_hard(α) = 0.4 * α` we *replace* (not augment) the random `AttackLayer` output with a deterministic resize-then-JPEG chain at one of three eval-mirrored operating points, drawn uniformly per batch. The remaining `1 - p_hard(α)` of post-clean batches use the unchanged `AttackLayer` (so easy and medium attacks stay in distribution and the encoder does not forget high-quality JPEG / mild-resize behavior).

The hard chain is constructed inside `baseline/exp0_inn_train.py` from `kornia.augmentation.RandomJPEG` and a copy of `infra.attacks.RandomDownUpResize`'s logic (re-implemented locally as `_DetDownUpResize` with `p=1.0`), so no `infra/` file is modified. The clean phase (`step < stage_a_steps`) is preserved verbatim — `attack(step)` returns `None` while `α==0`, so the secret pathway is established before any hard-chain pressure can collapse `bit_acc_clean`.

## 2. Files changed (Builder may only edit these)

- `code/baseline/exp0_inn_train.py` — extend `NoiseCurriculum` with the deterministic hard-chain bank and the `p_hard(α)` branch in `attack(step)`; add per-step logging of which branch fired.

Untouched: `code/baseline/inn_model.py`, `code/baseline/losses.py`, everything in `infra/**`.

## 3. CLI flag values for `baseline/exp0_inn_train.py` (config)

Locked (must NOT change): `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.

Tuned for this design (all baseline defaults retained):
- `--num_blocks 16`
- `--subnet_type rdb`
- `--lr 2e-4`
- `--img_weight 1.0`
- `--bit_weight 20.0`
- `--stage_a_steps 5000`
- `--stage_b_steps 10000`

Other flags keep baseline defaults (`--ckpt_every 2000`, `--log_every 200`, `--val_every 1000`, `--val_n 50`, `--num_workers 4`, `--amp` on, `--seed 0`).

No new CLI flag is added. The hard-chain probability scale `0.4` and the operating-point bank are hard-coded constants inside `NoiseCurriculum` (see §4); changing them is a separate design.

## 4. Algorithmic changes (concrete)

### 4.1 New helper module-level class `_DetDownUpResize` in `exp0_inn_train.py`

Add after the imports, before `NoiseCurriculum`:

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
```

### 4.2 New helper class `_HardChain` in `exp0_inn_train.py`

```python
import kornia.augmentation as K  # add to imports if not already present

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

### 4.3 Extend `NoiseCurriculum`

Modify `NoiseCurriculum.__init__` to also build the hard-chain bank:

```python
self._hard_chains = torch.nn.ModuleList([
    _HardChain(scale=0.5, q=85),  # eval chain_insta operating point
    _HardChain(scale=0.5, q=70),  # eval chain_whatsapp_std operating point
    _HardChain(scale=0.3, q=50),  # eval chain_wechat operating point
])
self._p_hard_max = 0.4   # asymptotic hard-chain firing probability at α=1
self.last_branch = "clean"  # diagnostic for logging
```

Update `to(device)` to also move the bank:

```python
def to(self, device):
    self._attack.to(device)
    self._hard_chains.to(device)
    return self
```

Replace `attack(step)` with the branched version:

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
```

`stage(step)` is unchanged (still returns `clean` / `ramp(α)` / `full`).

### 4.4 Sanity-check the new chains at startup

In `main()`, immediately after `curriculum._attack.sanity_check()`, add:

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

### 4.5 Logging additions

Extend the per-`log_every` `rec` dict in `metrics.jsonl` to include:
- `attack_branch` (str) — `curriculum.last_branch` at the moment the step ran.

Keep all existing fields unchanged. Update the printed line to append ` br=<branch>` so SLURM stdout is also informative.

In `val_metrics.jsonl`, no change (validation still uses `_val_build_attacks`).

## 5. Expected behavior after the change

- `metrics.jsonl` lines now include `attack_branch`. After step 15000, ~40% of attacked batches log a `hard[...]` branch; before step 5000, all batches log `clean`.
- `bit_acc_ema` may dip 0.01–0.03 for ~1k steps after `step=stage_a_steps` while the encoder absorbs the harder distribution, then recovers above baseline by step ~20000.
- `psnr_ema` at step 30000 is expected within ±0.3 dB of baseline (no loss-function change; only the input-noise distribution shifts).
- Eval `bit_acc` on `chain_wechat`, `chain_whatsapp_std`, `chain_x` should rise materially (target: `chain_wechat` ≥ 0.997 vs baseline 0.988); easier eval attacks stay ≥ 0.999.
- `bit_acc_clean = 1.000` at every eval resolution must hold post-eval.
- Wallclock: extra kornia.JPEG forward on ~40% of post-α batches; expected slowdown ≤ 8% over baseline. 30k steps fit comfortably in 12 h on a 2080Ti.
- `score = 2 * psnr_mean * (bit_acc_mean - 0.5)` is expected to improve over baseline 27.27 (motivation §"Score sensitivity": +0.5 to +0.7).

## 6. Constraints and invariants Builder must preserve

- Do NOT modify any `infra/**` file. The hard-chain implementation must live inside `code/baseline/exp0_inn_train.py`.
- Do NOT change locked CLI flags (`--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`).
- Do NOT raise `--bit_weight` above 20.0; this design must isolate the curriculum axis from the bit-pressure axis.
- Do NOT swap in `ImageSecretLoss` — loss formulation stays at pure-L1 host + BCE bits (idea002 explicitly forbids combining with idea001's loss-rebalancing axis).
- Preserve the clean phase (`step < stage_a_steps`): `attack(step)` MUST return `None` while `α == 0`. The hard-chain branch is gated on `α > 0` via `p_hard = p_hard_max * α`, so it cannot fire during the clean phase.
- Preserve gradient clip `clip_grad_norm_(..., 5.0)`, optimizer (AdamW, betas (0.9, 0.999), wd 0), cosine LR schedule, AMP setup, and the resume/checkpoint logic verbatim.
- Preserve `container = container.clamp(-1.0, 1.0)` before the attack call.
- Hard-chain operating points are fixed at `{(0.5, 85), (0.5, 70), (0.3, 50)}` — these mirror the eval suite's named chains. Builder must NOT add `(0.4, 75)` (chain_x) or any other point; that mirrors only the three named chains called out in the idea, and adding more is a different design.
- All edits must live inside `code/baseline/`; no edits anywhere else.

## 7. Edge cases

- Resume from `ckpt_latest.pt`: `_HardChain` and `_DetDownUpResize` carry no learnable parameters and no buffers requiring restore (kornia.RandomJPEG holds an internal LUT initialized in `__init__`); curriculum reconstruction at process start is sufficient.
- AMP: `_HardChain.forward` already casts to fp32 internally (mirroring `infra.attacks.AttackLayer.forward`), so it is safe to call from inside the autocast block.
- RNG determinism: the per-step `torch.rand(1)` call for `p_hard` and `torch.randint` for chain selection are drawn from the global torch RNG, whose state is already saved/restored by the existing checkpoint code (`torch_rng` / `cuda_rng` keys).
- If kornia raises on the startup probe, fail fast — do not try/except around the sanity check.
- If `stage_a_steps == 0` (clean phase disabled), `α` jumps to a positive value at step 0 and the hard branch can fire immediately. This design assumes the baseline default `stage_a_steps=5000` is used; the constraint above forbids changing it.
