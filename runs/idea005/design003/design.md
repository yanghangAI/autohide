**Design Description:** Walsh–Hadamard 1024-slot spreading — replace the bit→32×32-grid step in `PatchBitAdapter.encode` with `(B, 1024) → matmul by H_1024 / sqrt(1024) → reshape to (32, 32) → upsample to DWT plane`, so every output pixel is a sum over all 1024 sign-masked bits via orthogonal Hadamard rows. Decode pools to 32×32 then inverts with `H_1024^T / sqrt(1024)`. Zero-parameter, exactly orthogonal, exactly invertible on clean.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Goal

Eliminate the visible 32×32 grid by spreading each bit's energy uniformly over all 1024 grid cells via an orthogonal Hadamard basis. No bit lives in a single block: the per-cell carrier value is a balanced ±1 sum of all 1024 (sign-masked) bits, so the spatial pattern is broadband and structureless. This is the principled-basis × scramble corner of the 2×2 design family. Expectation: container PSNR rises (the carrier no longer aligns to a periodic grid), and `bit_acc_clean = 1.000` is preserved exactly because Hadamard with the `1/sqrt(1024)` scaling is orthonormal so its transpose is its inverse.

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

### 4.1 Add a Hadamard-matrix builder near the top of the file (right after `_make_sign_mask`, before `class PatchBitAdapter`)

```python
def _make_hadamard_orthonormal(n: int = NUM_SLOTS) -> torch.Tensor:
    """Sylvester-construction Hadamard matrix of size n×n, scaled to be
    orthonormal: H @ H.T == I exactly (for n a power of two).

    Requires n to be a power of two. For NUM_SLOTS = 1024 = 2**10 this
    holds. Returns a fp32 tensor of shape (n, n).
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of two; got {n}")
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.size(0) < n:
        H = torch.cat([torch.cat([H,  H], dim=1),
                        torch.cat([H, -H], dim=1)], dim=0)
    # Sylvester H_n has H_n @ H_n.T == n * I. Scale by 1/sqrt(n) for
    # orthonormality.
    return H / float(n) ** 0.5
```

### 4.2 Replace the body of `PatchBitAdapter` with the Hadamard variant

Keep the class name and method signatures (`__init__(self, n_bits: int = 896)`, `encode(bits, h_dwt, w_dwt)`, `decode(tensor_rec)`) exactly as today. Internally:

```python
class PatchBitAdapter(nn.Module):
    """Bit ↔ tensor adapter via orthonormal Walsh–Hadamard spreading.

    Forward (encode): bits (B, n_bits) → bipolar ±1 → pad to 1024 slots →
    sign-mask scramble → matmul by H (1024×1024 orthonormal Hadamard) →
    reshape to (B, 1, 32, 32) → nearest-neighbor upsample to (h_dwt, w_dwt)
    → broadcast to 12 channels.

    Reverse (decode): tensor (B, 12, h_dwt, w_dwt) → mean over 12 channels
    → adaptive_avg_pool2d to (32, 32) → flatten to (B, 1024) → matmul by
    H.T (the inverse of orthonormal H) → undo sign-mask → take first n_bits
    as logits.
    """

    def __init__(self, n_bits: int = 896):
        super().__init__()
        if n_bits > NUM_SLOTS:
            raise ValueError(f"n_bits={n_bits} > {NUM_SLOTS}")
        self.n_bits = n_bits
        sign = _make_sign_mask(NUM_SLOTS)
        self.register_buffer("sign_mask", sign, persistent=True)
        H = _make_hadamard_orthonormal(NUM_SLOTS)
        # Persist both H and H.T explicitly (pre-transposed for decode-time
        # speed and to make checkpoints self-contained without recomputing).
        self.register_buffer("hadamard", H, persistent=True)             # (1024, 1024)
        self.register_buffer("hadamard_t", H.t().contiguous(), persistent=True)

    def encode(self, bits: torch.Tensor, h_dwt: int, w_dwt: int) -> torch.Tensor:
        B = bits.size(0)
        bipolar = bits * 2.0 - 1.0
        full = bits.new_zeros((B, NUM_SLOTS))
        full[:, : self.n_bits] = bipolar
        scrambled = full * self.sign_mask                  # (B, 1024)
        spread = scrambled @ self.hadamard.to(scrambled.dtype)  # (B, 1024)
        grid = spread.view(B, 1, GRID, GRID)
        if h_dwt % GRID != 0 or w_dwt % GRID != 0:
            raise ValueError(
                f"DWT spatial dims ({h_dwt}, {w_dwt}) must each be divisible "
                f"by {GRID}."
            )
        rep_h = h_dwt // GRID
        rep_w = w_dwt // GRID
        tiled = grid.repeat_interleave(rep_h, dim=2).repeat_interleave(rep_w, dim=3)
        return tiled.expand(B, INN_CHANNELS, h_dwt, w_dwt).contiguous()

    def decode(self, tensor_rec: torch.Tensor) -> torch.Tensor:
        B = tensor_rec.size(0)
        pooled = F.adaptive_avg_pool2d(tensor_rec, (GRID, GRID))  # (B, 12, 32, 32)
        pooled = pooled.mean(dim=1)                              # (B, 32, 32)
        flat = pooled.reshape(B, NUM_SLOTS)
        # H is orthonormal so H^{-1} = H^T.
        unspread = flat @ self.hadamard_t.to(flat.dtype)         # (B, 1024)
        unscrambled = unspread * self.sign_mask
        return unscrambled[:, : self.n_bits]
```

### 4.3 Why the clean path is exactly invertible

- `scrambled` is the input to the encode-side linear map M_enc(x) = `tile(reshape(x @ H))` where `tile` is broadcast and nearest-neighbor upsampling.
- `decode` applies `mean over channels → adaptive_avg_pool2d → flatten → @ H^T → mask`. The compositions `tile → mean over channels → adaptive_avg_pool2d` give back exactly `reshape(x @ H)` on the clean path because:
  - `tile` produces a spatially constant 4×4 (or rep_h × rep_w) block per grid cell.
  - `adaptive_avg_pool2d(., (32, 32))` averages each block back to its constant value.
  - `mean over channels` is the identity since the carrier was broadcast across all 12 channels.
- Then `(x @ H) @ H^T = x @ (H H^T) = x @ I = x`, since `H` is orthonormal by construction.
- Multiplying by the sign-mask twice gives `x` again. So `decode(encode(bits)) = bits` (up to fp32 round-off ≪ 1).

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
        raise RuntimeError("PatchBitAdapter (Hadamard) clean-path "
                           "invertibility broken: sign mismatch")
    # Stronger orthonormality check (numeric tolerance):
    _err = (_rec - (_bits * 2.0 - 1.0)).abs().max().item()
    if _err > 1e-3:
        raise RuntimeError(
            f"PatchBitAdapter (Hadamard) clean-path numerical drift "
            f"too large: max |err| = {_err:.2e} > 1e-3"
        )
    print(f"[adapter] hadamard invertibility check: PASS (max err {_err:.2e})")
```

## 5. Logging spec

Add one diagnostic field to the per-step `rec` dict:

```python
adapter_kind="hadamard",
```

No other logging change. `metrics.jsonl` schema stays a strict superset of baseline.

## 6. Expected behaviour after the change

- **Forward cost:** one 1024×1024 matmul on (B, 1024) tensors per encode and per decode. At B=16 that's 16×1024×1024 ≈ 1.7e7 multiply-adds — well under 0.1 % of one DKiS block. Negligible.
- **Carrier appearance:** the 32×32 grid in DWT space contains a balanced ±1 pattern (since each grid cell is a sum of 1024 ±1 bits divided by sqrt(1024), the per-cell value is centered at 0 with stddev ≈ 1). After tile+iDWT, container_rgb's bit-carrier component is a broadband noise-like field with no periodic spike. Fourier spectrum is approximately flat across the 32×32 grid frequencies; periodicity at the 8-px-RGB frequency is gone.
- **`metrics.jsonl`** per-step keys = baseline keys ∪ `{adapter_kind}`.
- **`ckpt_latest.pt`** payload now includes new buffers `adapter.hadamard` (1024×1024 fp32 = 4 MiB) and `adapter.hadamard_t` (also 4 MiB), plus the unchanged `sign_mask`. Total ckpt size up by ~8 MiB; well within the GiB-scale storage budget.
- **Eval `metrics.csv`:** target `score >= 27.5` (≥ baseline + 0.25). Hard condition: `bit_acc_clean = 1.000` at every one of the 9 resolutions; structurally guaranteed by orthonormality.

## 7. Invariants Builder must preserve

1. **Locked flags unchanged:** `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.
2. **Locked infra unchanged:** zero edits under `infra/**`. `PatchBitAdapter.encode/decode` keep their public signatures.
3. **Hard condition:** `bit_acc_clean = 1.000` at every eval resolution. Structurally guaranteed by orthonormal H plus the matched mean-pool decode; step-0 sign + 1e-3 numeric assertion in §4.4 guards implementation slips.
4. **Determinism:** Hadamard is deterministic by Sylvester construction; sign-mask is the same fixed seed as baseline. Two runs with the same RNG seed produce byte-identical containers.
5. **Orthonormal scaling:** the `1/sqrt(1024)` scaling is essential — without it, `H @ H^T = 1024 * I` and decode would scale logits by 1024 and shift the BCE-with-logits operating point. Builder must compute H exactly as in §4.1 and not substitute a `±1` (unscaled) Hadamard.
6. **Resume correctness:** the new buffers `hadamard` and `hadamard_t` are persistent and round-trip via `state_dict`. They must be cast to the running dtype on the fly (the `.to(scrambled.dtype)` call in encode/decode) so AMP runs with fp16 activations don't break.
7. **No new CLI flags.**

## 8. Edge cases

- **Bit-carrier amplitude:** at full 1024 bits, each grid cell value is a sum of 1024 ±1 / sqrt(1024) terms — distribution N(0, 1) in the limit. With 896 bits + 128 zero pads, the variance is ≈ 896/1024 ≈ 0.875. The INN sees a similar-magnitude carrier as before (stddev ~1 vs. baseline's exact ±1), so loss-weight tuning is not needed.
- **Eval at smaller resolutions (e.g. 64-px RGB → 32×32 DWT):** `repeat_interleave(rep_h=1, rep_w=1)` is a no-op; carrier already at 32×32. Decode `adaptive_avg_pool2d` to 32×32 of a 32×32 input is a no-op. Hadamard round-trip still applies and is sign-perfect.
- **Eval at larger resolutions (e.g. 1024-px RGB → 512×512 DWT):** rep_h = rep_w = 16; the 32×32 Hadamard-transformed grid is upsampled by 16×; decode pools back to 32×32 and inverts. Sign-perfect on clean.
- **fp16 / autocast:** the explicit `.to(dtype)` cast on the Hadamard buffer keeps the matmul inside autocast at fp16 (4 MiB cast per call, cheap). Numeric error scales as O(sqrt(1024)) × fp16 ulp ≈ 1e-2 — well under the slot magnitude of ~1, so sign recovery is preserved. The 1e-3 max-error gate in §4.4 is run in fp32 (outside autocast) to avoid spurious fp16-driven failures.
- **Memory:** the 8 MiB extra buffer is permanently resident on GPU; trivial vs. activation memory at batch 16.
