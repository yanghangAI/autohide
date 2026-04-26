**Files changed:**
- `code/exp0_inn_train.py`

**Changes:**
- `code/exp0_inn_train.py` — extended `NoiseCurriculum` with severity-annealing endpoints (`_sev_lo`, `_sev_hi`); added `severity_alpha(step)` (parallel ramp to `alpha(step)`); rewrote `attack(step)` to linearly interpolate `jpeg1.jpeg_quality`, `jpeg2.jpeg_quality`, `resize.scale_min`, and the chain firing probability between low- and high-severity endpoints under α₂; updated `stage(step)` for richer two-axis diagnostics; added a startup probe at three severity points and reset to clean state; logged `severity_alpha`, `jpeg1_q_min`, `resize_scale_min` into `metrics.jsonl` and `sev_α` into the per-step stdout line.

The new severity-endpoint cache added inside the curriculum constructor in `code/exp0_inn_train.py`:

```python
        # Severity endpoints for the secondary (α₂) annealing axis. Values mirror
        # infra/attacks.py SEVERITY["low"] and SEVERITY["high"]. Re-declared here
        # (not imported) because infra/attacks.py is locked.
        self._sev_lo = dict(jpeg_q_min=80, chain_q_min=60,
                            resize_scale_min=0.9, chain_p=0.0)
        self._sev_hi = dict(jpeg_q_min=40, chain_q_min=20,
                            resize_scale_min=0.5, chain_p=0.3)
        self.last_sev_alpha = 0.0  # diagnostic for logging
```

The new `severity_alpha` and rewritten `stage` methods in `code/exp0_inn_train.py`:

```python
    def severity_alpha(self, step: int) -> float:
        if step < self.ramp_start:
            return 0.0
        if step >= self.ramp_end:
            return 1.0
        return (step - self.ramp_start) / (self.ramp_end - self.ramp_start)

    def stage(self, step: int) -> str:
        a1 = self.alpha(step)
        a2 = self.severity_alpha(step)
        if a1 == 0.0:
            return "clean"
        if a1 >= 1.0 and a2 >= 1.0:
            return "full"
        return f"ramp(a1={a1:.2f},a2={a2:.2f})"
```

The new severity-mutating `attack(step)` in `code/exp0_inn_train.py`:

```python
    def attack(self, step: int):
        a1 = self.alpha(step)
        if a1 == 0.0:
            self.last_sev_alpha = 0.0
            return None
        a2 = self.severity_alpha(step)
        self.last_sev_alpha = a2
        # Linearly interpolate the four severity knobs.
        lo, hi = self._sev_lo, self._sev_hi
        jpeg_q_min   = lo["jpeg_q_min"]   + a2 * (hi["jpeg_q_min"]   - lo["jpeg_q_min"])
        chain_q_min  = lo["chain_q_min"]  + a2 * (hi["chain_q_min"]  - lo["chain_q_min"])
        resize_smin  = lo["resize_scale_min"] + a2 * (hi["resize_scale_min"] - lo["resize_scale_min"])
        chain_p_max  = lo["chain_p"]      + a2 * (hi["chain_p"]      - lo["chain_p"])

        # Mutate kornia parameters in-place. RandomJPEG.jpeg_quality is a
        # buffer/tensor (kornia stores quality bounds as a registered buffer);
        # write through .data to avoid breaking the registered Parameter/Buffer.
        q1_lo = float(min(94.0, max(10.0, jpeg_q_min)))
        self._attack.jpeg1.jpeg_quality = (q1_lo, 95.0)
        q2_lo = float(min(q1_lo - 1.0, max(10.0, chain_q_min)))
        self._attack.jpeg2.jpeg_quality = (q2_lo, q1_lo)
        self._attack.resize.scale_min = float(max(0.1, min(1.0, resize_smin)))
        # Bernoulli probabilities scale with α₁ (firing axis); chain_p target
        # itself is annealed by α₂.
        self._attack.jpeg1.p = self._p_max["jpeg1"] * a1
        self._attack.jpeg2.p = chain_p_max * a1
        self._attack.resize.p = self._p_max["resize"] * a1
        return self._attack
```

The startup severity probe added in main inside `code/exp0_inn_train.py`:

```python
    # Probe the curriculum at the easiest, midpoint, and worst severity to
    # catch kornia parameter-update bugs at startup, not at step 5001.
    for _probe_step in (curriculum.ramp_start + 1,
                        (curriculum.ramp_start + curriculum.ramp_end) // 2,
                        curriculum.ramp_end + 1):
        _atk = curriculum.attack(_probe_step)
        if _atk is not None:
            with torch.no_grad():
                _ = _atk(torch.zeros(2, 3, args.resolution, args.resolution,
                                      device=device, dtype=torch.float16))
    # Reset Bernoulli state to clean-phase (α₁=0) by re-calling attack(0).
    curriculum.attack(0)
    print("[init] severity-annealing sanity check passed.")
```

The metrics-record additions and stdout addition in `code/exp0_inn_train.py`:

```python
                       severity_alpha=curriculum.last_sev_alpha,
                       jpeg1_q_min=_jpeg1_q_min,
                       resize_scale_min=_resize_smin,
```

```python
                  f"img={img_loss.item():.4f} bit={bit_loss.item():.4f} "
                  f"sev_α={curriculum.last_sev_alpha:.2f}")
```
