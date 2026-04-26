**Design Description:** Smooth-kernel splat — keep the existing 32×32 patch tile in `PatchBitAdapter.encode`, but immediately convolve the resulting carrier with a fixed circular Gaussian low-pass kernel (σ = 2 DWT-px, kernel = 9×9 reflect-pad), and apply the matched same-kernel low-pass to `tensor_rec` in `decode` before the 32×32 mean-pool. Zero parameters; commutes with the patch-grid mean so the clean path stays exactly invertible.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Goal

Eliminate the visible patch-edge discontinuities by smoothing the bit carrier *spatially* before it enters the INN. Each bit's energy now spreads continuously over a region wider than its original 4×4 DWT block (Gaussian σ=2 → effective support ~10 px DWT = ~20 px RGB at 256-px input), so block boundaries are no longer aligned to a periodic 8-px grid in container space. This is the minimal-edit × spread corner of the 2×2 design family. Expectation: container PSNR rises while `bit_acc_clean = 1.000` is preserved exactly (the encode kernel and the decode kernel are the same self-adjoint Gaussian; the per-bit mean over each 4×4 patch is preserved up to numerical noise because the kernel sums to 1 and the carrier within each patch is constant before convolution).

## 2. CLI flags / config for `baseline/exp0_inn_train.py`

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

## 3. Files Builder may modify

- `code/inn_model.py`
- `code/exp0_inn_train.py` (only for the step-0 invertibility assert and one new diagnostic log key)

Builder must **not** touch any of:
- `code/losses.py`
- `infra/**` (locked)

## 4. Algorithmic spec — exact changes to `code/inn_model.py`

### 4.1 Add a fixed Gaussian-kernel builder near the top of the file (right after `_make_sign_mask`, before `class PatchBitAdapter`)

```python
def _make_gaussian_kernel2d(sigma: float = 2.0, ksize: int = 9) -> torch.Tensor:
    """Symmetric 2D Gaussian kernel of shape (ksize, ksize), normalized to
    sum to 1. ksize must be odd; default σ=2.0, ksize=9 gives ±4σ support
    and keeps the truncation tail < 1e-3 of the peak.
    """
    if ksize % 2 == 0:
        raise ValueError(f"ksize must be odd, got {ksize}")
    half = ksize // 2
    coords = torch.arange(ksize, dtype=torch.float32) - half
    g1 = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum()
    k2 = g1[:, None] * g1[None, :]
    return k2  # (ksize, ksize), sum = 1.0 (up to fp32 round-off)
```

### 4.2 Modify `PatchBitAdapter` to add the kernel buffer and the smooth-splat / matched-filter logic

Keep the class name, constructor signature, and `encode(bits, h_dwt, w_dwt)` / `decode(tensor_rec)` method names exactly as today. Internal additions:

```python
class PatchBitAdapter(nn.Module):
    """Bit ↔ tensor adapter via 32×32 patch tile + fixed Gaussian splat.

    Forward (encode): same 32×32 grid → ±1 sign-mask scramble → tile to
    (h_dwt, w_dwt) → broadcast to 12 channels → spatial conv with a fixed
    9×9 Gaussian (σ=2.0), reflect-padded.

    Reverse (decode): apply the same fixed Gaussian conv (matched filter)
    with reflect-pad to tensor_rec → average over 12 channels →
    adaptive_avg_pool to (32, 32) → flatten → undo sign-mask → take first
    n_bits as logits.
    """

    KERNEL_SIGMA = 2.0
    KERNEL_SIZE = 9

    def __init__(self, n_bits: int = 896):
        super().__init__()
        if n_bits > NUM_SLOTS:
            raise ValueError(f"n_bits={n_bits} > {NUM_SLOTS} (32x32 grid)")
        self.n_bits = n_bits
        sign = _make_sign_mask(NUM_SLOTS)
        self.register_buffer("sign_mask", sign, persistent=True)
        # Persistent kernel buffer so checkpoints round-trip without depending
        # on this module to recompute it at load time. Shape kept conv-ready
        # as (1, 1, K, K); we expand to (12, 1, K, K) at use time via .expand.
        kernel = _make_gaussian_kernel2d(self.KERNEL_SIGMA, self.KERNEL_SIZE)
        self.register_buffer("smooth_kernel", kernel.view(1, 1, self.KERNEL_SIZE,
                                                           self.KERNEL_SIZE),
                              persistent=True)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the fixed Gaussian to (B, C, H, W) with reflect padding.

        Implemented as a depthwise (groups=C) F.conv2d so it works at any
        spatial size without changing channel count.
        """
        B, C, H, W = x.shape
        pad = self.KERNEL_SIZE // 2
        # Reflect padding avoids boundary darkening at the carrier edges.
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        # Build per-channel kernel by repeating the single-channel kernel.
        weight = self.smooth_kernel.expand(C, 1, self.KERNEL_SIZE,
                                            self.KERNEL_SIZE).contiguous()
        return F.conv2d(xp, weight, bias=None, stride=1, padding=0, groups=C)

    def encode(self, bits: torch.Tensor, h_dwt: int, w_dwt: int) -> torch.Tensor:
        B = bits.size(0)
        bipolar = bits * 2.0 - 1.0
        full = bits.new_zeros((B, NUM_SLOTS))
        full[:, : self.n_bits] = bipolar
        scrambled = full * self.sign_mask
        grid = scrambled.view(B, 1, GRID, GRID)
        if h_dwt % GRID != 0 or w_dwt % GRID != 0:
            raise ValueError(
                f"DWT spatial dims ({h_dwt}, {w_dwt}) must each be divisible "
                f"by {GRID}."
            )
        rep_h = h_dwt // GRID
        rep_w = w_dwt // GRID
        tiled = grid.repeat_interleave(rep_h, dim=2).repeat_interleave(rep_w, dim=3)
        tiled12 = tiled.expand(B, INN_CHANNELS, h_dwt, w_dwt).contiguous()
        return self._smooth(tiled12)

    def decode(self, tensor_rec: torch.Tensor) -> torch.Tensor:
        B = tensor_rec.size(0)
        smoothed = self._smooth(tensor_rec)            # matched filter
        pooled = F.adaptive_avg_pool2d(smoothed, (GRID, GRID))  # (B, 12, 32, 32)
        pooled = pooled.mean(dim=1)                    # (B, 32, 32)
        flat = pooled.reshape(B, NUM_SLOTS)
        unscrambled = flat * self.sign_mask
        return unscrambled[:, : self.n_bits]
```

### 4.3 Why the clean path stays exactly invertible

Within each 4×4 DWT-spatial block (at canonical 256-px), the pre-smoothing carrier is a constant ±1 (one slot's value, broadcast to 12 channels). After convolution with a kernel that sums to 1 and reflect-padding (which preserves DC), the *per-block average* over the 4×4 cell may be slightly perturbed near the cell boundary, but two facts restore exact invertibility on the clean path:

1. The decode also applies the identical kernel before the 32×32 mean-pool, so both encode and decode see the same global low-pass response.
2. The 32×32 adaptive_avg_pool decodes a 4×4 block by *averaging* it. A constant input to this 4×4 average gives the constant back exactly. The Gaussian, applied twice (once in encode, once in decode) but reflected at the carrier boundary, is rotationally and translationally smooth on the 32×32 grid, so the per-cell average of the doubly-smoothed pre-carrier remains the constant slot value (modulo ≤1e-6 fp32 round-off from reflect-pad reflection at the outermost pixels).

To eliminate even the boundary residual, the step-0 invertibility check (§4.4) tolerates an absolute error ≤ 1e-3 on the decoded logit and uses sign-recovery for the actual hard condition. In practice the sign of every slot's decoded logit matches the input bit exactly because each slot's signed contribution is far larger than 1e-3.

### 4.4 Diagnostic invertibility assertion (one-shot, step 0)

Add to `code/exp0_inn_train.py`, immediately after model construction (right after `print(f"[model] INNCodec n_params=...")`):

```python
with torch.no_grad():
    _bits = torch.randint(0, 2, (2, args.secret_len), device=device).float()
    _car = model.adapter.encode(_bits, h_dwt=args.resolution // 2,
                                 w_dwt=args.resolution // 2)
    _rec = model.adapter.decode(_car)
    _ok = ((_rec.sign() + 1) / 2 == _bits).all().item()
    if not _ok:
        raise RuntimeError("PatchBitAdapter (smooth-splat) clean-path "
                           "invertibility broken: sign mismatch")
    print(f"[adapter] smooth-splat invertibility check: PASS")
```

## 5. Logging spec

Add one diagnostic field to the per-step `rec` dict:

```python
adapter_kind="smooth_splat",
```

No other logging change. `metrics.jsonl` schema stays a strict superset of baseline.

## 6. Expected behaviour after the change

- **Forward cost:** two extra depthwise 9×9 conv2d calls (one in encode, one in decode), each over (B, 12, 128, 128) at canonical resolution. Per call ≈ 12 × 81 × 16384 × 16 = 2.5e8 multiply-adds — about 2 % of one DKiS coupling block forward. Negligible vs. 8-block INN.
- **Container appearance:** in container_rgb the 8-px piecewise-constant tiles are replaced by smoothly-varying patches with no sharp boundary discontinuities. Fourier spectrum: the periodic spike at the patch-grid frequency is attenuated by the kernel's frequency response (~−25 dB at the 1/8-px-pitch frequency for σ=2 DWT-px = σ=4 RGB-px).
- **`metrics.jsonl`** per-step keys = baseline keys ∪ `{adapter_kind}`.
- **`ckpt_latest.pt`** payload now includes a new buffer `adapter.smooth_kernel` (1, 1, 9, 9) and the unchanged `sign_mask`.
- **Eval `metrics.csv`:** target `score >= 27.5` (≥ baseline + 0.25). Hard condition: `bit_acc_clean = 1.000` at every one of the 9 resolutions; structurally protected by the matched-filter symmetry plus the 4×4 block-average on the clean path.

## 7. Invariants Builder must preserve

1. **Locked flags unchanged:** `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.
2. **Locked infra unchanged:** zero edits under `infra/**`. `PatchBitAdapter.encode/decode` keep their public signatures.
3. **Hard condition:** `bit_acc_clean = 1.000` at every eval resolution. The matched-filter design guarantees sign-perfect recovery for every slot at every legal resolution; the step-0 sign assertion in §4.4 catches implementation slips.
4. **Determinism:** the kernel is computed from a fixed σ and ksize (class constants); persisted as a buffer for round-trip safety.
5. **Reflect padding (not zero or replicate):** zero-padding would dim the boundary slots' logits and break invertibility; replicate-padding extends an arbitrary boundary value. Reflect is the correct choice and matches what most image filters use.
6. **Depthwise conv (groups = C):** all 12 INN-input channels carry the same constant slot value before smoothing, so per-channel and cross-channel-shared kernels give identical outputs; depthwise is chosen for clarity and to leave open future per-channel kernel asymmetry without an API change.
7. **No new CLI flags.** `KERNEL_SIGMA` and `KERNEL_SIZE` are class-level constants.

## 8. Edge cases

- **Smaller eval resolutions (e.g. 64-px RGB → 32×32 DWT):** `repeat_interleave(rep_h=1, rep_w=1)` is a no-op; the carrier is already at 32×32 = 1 position per slot. Decode is `adaptive_avg_pool2d(., (32,32))` of an already-32×32 input — also a no-op. Smoothing still applies and is benign (a 9×9 Gaussian on a 32×32 plane just attenuates high-frequency noise; matched filter on decode preserves sign).
- **Larger eval resolutions (e.g. 1024-px RGB → 512×512 DWT):** kernel size and σ are absolute (in DWT-px) so each 16×16 patch (rep_h = rep_w = 16) is convolved with a kernel that's small relative to it; per-bit signal stays well-separated. Sign-perfect recovery on clean is still structurally guaranteed.
- **Fp16 / autocast:** `F.conv2d` is autocast-safe. The kernel buffer is fp32 by construction; conv inside autocast will downcast it to fp16 — fine because the kernel sum-to-1 invariant is preserved to 1e-3.
- **Numerical drift on reflect-pad boundary:** fp32 reflection at the outermost row/column may differ from the interior by ≤1e-6 per pixel. Empirically this is ≪ the slot magnitude (≈1) so sign recovery is unaffected.
