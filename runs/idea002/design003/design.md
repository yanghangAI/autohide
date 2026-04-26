**Design Description:** Per-attack hard-example mining: every K=500 steps, score a fixed bank of 5 chains on a held-out batch by current `bit_acc`, then bias the per-step training-attack sampler toward the worst-performing chain via softmax-temperature reweighting; concentrates compute on whichever attack is currently the bottleneck.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Motivation and approach

The baseline curriculum samples training attacks i.i.d. from `AttackLayer(severity="high")`, allocating roughly equal compute to all (resize, jpeg, chain) combinations regardless of which one is currently the bottleneck. Per `idea.md` motivation, the wechat tail (`bit_acc=0.988`) is a clear outlier while the rest is ≥0.999 — i.e. >99% of training compute is being spent on attacks the encoder has already mastered.

This design adds a small, fixed bank of 5 deterministic chains (mirroring the eval suite's worst chains plus the existing random `AttackLayer`), and re-weights the per-step training sampler over that bank by current measured `bit_acc` on a held-out batch. Every K=500 steps, while the model is in eval mode, we score each chain on a fixed 16-image probe; the per-chain sampling weight is `softmax((1 - bit_acc_chain) / τ)` with temperature `τ = 0.05`. Bank composition is fixed; only the sampling distribution moves. This guarantees the encoder spends most of its post-α gradient updates on whichever attack is currently below its peers, automatically following the bottleneck as it shifts (initially wechat; possibly something else after wechat is closed).

The probe batch is constructed once at startup from the held-out `--val_root` (same source the existing inline-validation uses), so no new data path is introduced. Construction of the bank's chains is identical in shape to design001 (`_HardChain` instances + the existing `AttackLayer` as the "random" entry), built locally in `baseline/exp0_inn_train.py` from kornia primitives.

## 2. Files changed (Builder may only edit these)

- `code/baseline/exp0_inn_train.py` — extend `NoiseCurriculum` with the chain bank, the periodic probe, the bit-acc-weighted sampler, and logging.

Untouched: `code/baseline/inn_model.py`, `code/baseline/losses.py`, everything in `infra/**`.

## 3. CLI flag values for `baseline/exp0_inn_train.py` (config)

Locked: `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.

Tuned for this design (all baseline defaults retained):
- `--num_blocks 16`
- `--subnet_type rdb`
- `--lr 2e-4`
- `--img_weight 1.0`
- `--bit_weight 20.0`
- `--stage_a_steps 5000`
- `--stage_b_steps 10000`

Other flags: baseline defaults (`--ckpt_every 2000`, `--log_every 200`, `--val_every 1000`, `--val_n 50`, `--num_workers 4`, `--amp` on, `--seed 0`).

The mining hyperparameters are hard-coded constants: probe interval `K=500` steps, probe batch size = 16 (one batch from `--val_root`), softmax temperature `τ=0.05`, minimum per-chain weight floor `0.05` (so no chain is starved entirely). No new CLI flag is added.

## 4. Algorithmic changes (concrete)

### 4.1 Bank construction in `NoiseCurriculum.__init__`

Reuse the `_DetDownUpResize` and `_HardChain` helper-class definitions from design001 (Builder MUST copy them verbatim into this file; they are not imported from elsewhere because the design parents on `baseline/` directly):

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

In `NoiseCurriculum.__init__`, after the existing `self._p_max = {...}` block:

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

Update `to(device)` to also push the deterministic chains:

```python
def to(self, device):
    self._attack.to(device)
    for _, m in self._bank[1:]:  # skip the random entry, already moved above
        m.to(device)
    self._bank_weights = self._bank_weights.to(device)
    self._bank_bit_acc = self._bank_bit_acc.to(device)
    return self
```

### 4.2 New `attack(step)` that samples from the bank

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
```

### 4.3 Mining update method

Add to `NoiseCurriculum`:

```python
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
```

### 4.4 Probe-batch construction in `main()`

After the `val_paths` block in `main()`:

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

### 4.5 Periodic mining call in the training loop

After the existing `if global_step % args.ckpt_every == 0: save_ckpt()` line, add:

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

### 4.6 Logging additions

Extend the per-`log_every` `rec` dict in `metrics.jsonl` to include:
- `attack_branch` (str) — `curriculum.last_branch`.
- `bank_idx` (int) — `curriculum.last_bank_idx`.
- `bank_weights` (list[float]) — `curriculum._bank_weights.tolist()`.

Update the printed line to append ` br=<branch>`.

`val_metrics.jsonl`: unchanged.

### 4.7 Sanity-check the bank at startup

In `main()`, after `curriculum._attack.sanity_check()`, add:

```python
_probe = torch.zeros(2, 3, args.resolution, args.resolution, device=device, dtype=torch.float16)
_probe[0, 0, 0, 0] = float("nan")
_probe[0, 0, 0, 1] = float("inf")
with torch.no_grad():
    for name, mod in curriculum._bank[1:]:
        _ = mod(_probe)
print(f"[init] bank sanity check passed: {[n for n, _ in curriculum._bank]}")
```

## 5. Expected behavior after the change

- `metrics.jsonl` lines now include `attack_branch`, `bank_idx`, `bank_weights`.
- A new `[mining ...]` line is printed every 500 steps (after `step >= stage_a_steps`) showing the latest weights and per-chain bit_acc.
- Early in the ramp, weights are roughly uniform; once one chain (expected: `hard_s0.3_q50`) lags the others, its weight rises toward ~0.6–0.8 and the others fall to the floor (0.05). As the encoder closes that gap, weights re-equilibrate.
- `bit_acc_ema` should track baseline through the ramp and exceed baseline by step ~20000.
- `psnr_ema` at step 30000 expected within ±0.3 dB of baseline.
- Eval `bit_acc` on `chain_wechat`, `chain_whatsapp_std`, `chain_x` should rise (target: `chain_wechat` ≥ 0.997 vs baseline 0.988); easier eval attacks stay ≥ 0.999.
- `bit_acc_clean = 1.000` at every eval resolution must hold post-eval.
- Wallclock: probe runs every 500 steps with batch 16 over 5 chains in eval mode (~0.5 s); over 30k steps, ~30 s of overhead total. Per-step cost unchanged.
- `score = 2 * psnr_mean * (bit_acc_mean - 0.5)` is expected to improve over baseline 27.27.

## 6. Constraints and invariants Builder must preserve

- Do NOT modify any `infra/**` file. The bank chains are constructed in `code/baseline/exp0_inn_train.py` only.
- Do NOT change locked CLI flags (`--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`).
- Do NOT raise `--bit_weight` above 20.0; isolation from the bit-pressure axis is required.
- Do NOT swap in `ImageSecretLoss` — loss stays pure-L1 host + BCE bits (idea002 forbids combining with idea001's axis).
- Preserve the clean phase: `attack(step)` MUST return `None` while `α == 0`. The mining update MUST NOT fire before `step >= curriculum.ramp_start`.
- Mining MUST run with `model.eval()` and inside `torch.no_grad()`; gradients on the probe MUST NOT contaminate the optimizer state. Restore `model.train()` after the update.
- Bank composition is FIXED at the 5 entries listed in §4.1 (one random + four deterministic). Builder must NOT add or remove entries; that is a different design.
- Probe batch is FIXED at startup. Builder must NOT regenerate it during training (would inject an extra source of noise into the weight update).
- Per-chain weight floor is FIXED at 0.05 so no chain is permanently starved.
- Preserve gradient clip, optimizer, cosine LR schedule, AMP setup, and resume/checkpoint logic verbatim.
- Preserve `container = container.clamp(-1.0, 1.0)` before the attack call in the main loop.
- All edits must live inside `code/baseline/`; no edits anywhere else.

## 7. Edge cases

- Resume from `ckpt_latest.pt`: `_bank_weights` is currently NOT in the checkpoint. Builder MUST add `bank_weights` to the `payload` saved by `save_ckpt()` (cast to a list of floats) and restore it in the resume block, defaulting to uniform if the key is missing (backward-compat with pre-design001 checkpoints if any).
- If `--val_root` is empty (mining_probe is None), the loop falls back to the initial uniform `_bank_weights`; no mining update occurs but training proceeds with the bank-based sampler. This is acceptable degraded behavior, not a failure.
- AMP: `update_weights` runs under `torch.no_grad()` outside the autocast block. The internal forward through `model.embed` / `model.extract` is fp32; this is fine for the held-out probe.
- RNG determinism: `torch.multinomial` and `torch.randint` calls use the global torch RNG, already snapshotted in checkpoints.
- The `random` bank entry mutates `self._attack.jpeg{1,2}.p` and `self._attack.resize.p` — Builder must ensure these mutations are confined to the call (every other call into the same instance reassigns them), so there is no cross-call leakage.
- If `len(val_paths) < args.batch_size`, the probe uses fewer images. With `--val_n 50` and `--batch_size 16` this is not a concern.
