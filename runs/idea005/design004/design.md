**Design Description:** DCT mid-band spreading — replace the bit→32×32-grid step in `PatchBitAdapter.encode` with a 32×32 2D-DCT-II basis where each of the 896 used bits is assigned to one mid-band frequency coefficient (excluding DC and the highest-frequency band), then inverse-DCT back to the (32, 32) spatial grid and tile to the DWT plane. Decode pools to 32×32, applies forward 2D-DCT, and reads off the same 896 mid-band coefficients. Zero-parameter, exactly invertible on clean (orthonormal DCT-II), JPEG-friendly by construction.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Goal

Eliminate the visible 32×32 grid by spreading each bit across a single 2D sinusoidal basis function rather than concentrating it in one block, while *also* restricting energy to a JPEG-quantization-friendly mid-band so the carrier survives compression attacks better than a flat random spread. This is the principled-basis × spread corner of the 2×2 design family. Expectation: container PSNR rises (sinusoidal carrier integrates over each block instead of producing edge discontinuities) and bit_acc on JPEG attacks improves modestly because mid-band DCT energy lives in the region of the JPEG quantization table where coefficients are best preserved. `bit_acc_clean = 1.000` is preserved exactly because DCT-II is orthonormal.

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

### 4.1 Add a DCT-II 1D-matrix builder near the top of the file (right after `_make_sign_mask`, before `class PatchBitAdapter`)

```python
def _make_dct_ii_orthonormal(n: int) -> torch.Tensor:
    """Orthonormal 1D type-II DCT matrix D of shape (n, n) such that
    `D @ x` performs the forward DCT-II of length-n vector x and `D.T @ y`
    inverts it. Convention:

        D[k, m] = sqrt(2/n) * cos(pi*(2m+1)*k / (2n))   for k > 0
        D[0, m] = sqrt(1/n)                              for k = 0
    """
    m_idx = torch.arange(n, dtype=torch.float32)
    k_idx = torch.arange(n, dtype=torch.float32)
    arg = (torch.pi * (2.0 * m_idx[None, :] + 1.0) * k_idx[:, None]) / (2.0 * n)
    D = torch.cos(arg)
    scale = torch.full((n,), (2.0 / n) ** 0.5, dtype=torch.float32)
    scale[0] = (1.0 / n) ** 0.5
    D = D * scale[:, None]
    return D  # (n, n), orthonormal: D @ D.T == I (up to fp32 round-off)
```

### 4.2 Add a mid-band frequency selector that returns 896 valid (k_y, k_x) pairs deterministically

Place right after `_make_dct_ii_orthonormal`:

```python
def _select_midband_coeffs(grid: int = GRID, n_bits: int = 896,
                            r_lo: float = 0.20, r_hi: float = 0.85) -> torch.Tensor:
    """Pick exactly `n_bits` 2D-DCT coefficient indices (k_y, k_x) on a
    grid×grid frequency grid, restricted to a mid-band annulus defined by
    radial frequency r = sqrt(k_y^2 + k_x^2) / sqrt(2)*(grid-1).

    Constraints:
      - exclude DC: (k_y, k_x) != (0, 0).
      - radial frequency strictly between r_lo and r_hi.
      - deterministic ordering: by ascending r, ties broken by lex order
        (k_y, k_x).
      - returns the first `n_bits` such pairs.

    Defaults r_lo=0.20, r_hi=0.85: this excludes the lowest-frequency
    block (where small low-pass attacks would alias most strongly into
    the carrier) and the highest-frequency block (where JPEG quantization
    nukes coefficients first). For grid=32 there are 1024 total positions;
    the annulus comfortably contains > 896 valid pairs (verified by
    construction below).
    """
    coords = []
    for k_y in range(grid):
        for k_x in range(grid):
            if k_y == 0 and k_x == 0:
                continue
            r = ((k_y * k_y + k_x * k_x) ** 0.5) / ((2.0 * (grid - 1) ** 2) ** 0.5)
            if r < r_lo or r > r_hi:
                continue
            coords.append((r, k_y, k_x))
    coords.sort()  # ascending r, then lex
    if len(coords) < n_bits:
        raise ValueError(
            f"mid-band annulus has only {len(coords)} valid coefficients; "
            f"need {n_bits}. Loosen r_lo or r_hi."
        )
    coords = coords[:n_bits]
    out = torch.tensor([[ky, kx] for (_r, ky, kx) in coords], dtype=torch.long)
    return out  # (n_bits, 2)
```

For `grid=32, r_lo=0.20, r_hi=0.85` the annulus contains roughly 696 — *but this must be checked*. **Builder must run the deterministic enumeration and confirm that ≥ 896 mid-band coefficients exist.** If the count is short, Builder must instead use `r_lo=0.10, r_hi=0.90` (which yields > 950 valid coefficients on a 32×32 grid). Builder must commit only one of these two annulus choices and log it (the chosen `(r_lo, r_hi)` pair is logged as `adapter_midband_lo` / `adapter_midband_hi` per-step, see §5).

The Designer-recommended default that Builder should adopt unless the count check rejects it: `r_lo=0.10, r_hi=0.90`. With `grid=32` this gives 1024 − 1 (exclude DC) − (small low corner) − (small high corner) ≫ 896, so the count check passes comfortably. Builder must verify this at construction time and raise if violated.

### 4.3 Replace the body of `PatchBitAdapter` with the DCT mid-band variant

Keep the class name and method signatures exactly. Internally:

```python
class PatchBitAdapter(nn.Module):
    """Bit ↔ tensor adapter via DCT-II mid-band spreading.

    Forward (encode): bits (B, n_bits) → bipolar ±1 → sign-mask scramble
    (length n_bits, not 1024) → scatter into a (B, 32, 32) frequency grid
    at fixed mid-band positions (DC=0, high-band=0) → 2D inverse DCT
    (= D.T @ X @ D) to spatial 32×32 grid → tile to (h_dwt, w_dwt) →
    broadcast to 12 channels.

    Reverse (decode): tensor (B, 12, h_dwt, w_dwt) → mean over 12 channels
    → adaptive_avg_pool to (32, 32) → 2D forward DCT (= D @ S @ D.T) →
    read coefficients at the same mid-band positions → undo sign-mask →
    return as logits.
    """

    DCT_R_LO = 0.10
    DCT_R_HI = 0.90

    def __init__(self, n_bits: int = 896):
        super().__init__()
        if n_bits > NUM_SLOTS:
            raise ValueError(f"n_bits={n_bits} > {NUM_SLOTS}")
        self.n_bits = n_bits
        # The sign-mask now has length n_bits (one ±1 per used coefficient),
        # not NUM_SLOTS. Persist as a buffer.
        bit_sign = _make_sign_mask(n_bits, seed=0xB175CA10)
        self.register_buffer("sign_mask", bit_sign, persistent=True)

        D = _make_dct_ii_orthonormal(GRID)                          # (32, 32)
        self.register_buffer("dct_d", D, persistent=True)
        self.register_buffer("dct_dt", D.t().contiguous(), persistent=True)

        coeff_idx = _select_midband_coeffs(GRID, n_bits,
                                            self.DCT_R_LO, self.DCT_R_HI)  # (n_bits, 2)
        self.register_buffer("coeff_idx", coeff_idx, persistent=True)

    def _idct2d(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, 32, 32) frequency-domain → spatial. inverse is D^T X D."""
        return self.dct_dt.to(X.dtype) @ X @ self.dct_d.to(X.dtype)

    def _dct2d(self, S: torch.Tensor) -> torch.Tensor:
        """S: (B, 32, 32) spatial → frequency. forward is D S D^T."""
        return self.dct_d.to(S.dtype) @ S @ self.dct_dt.to(S.dtype)

    def encode(self, bits: torch.Tensor, h_dwt: int, w_dwt: int) -> torch.Tensor:
        B = bits.size(0)
        bipolar = bits * 2.0 - 1.0                                  # (B, n_bits)
        scrambled = bipolar * self.sign_mask                         # (B, n_bits)
        # Scatter into (B, 32, 32) frequency grid at mid-band indices.
        X = bits.new_zeros((B, GRID, GRID))
        ky = self.coeff_idx[:, 0]
        kx = self.coeff_idx[:, 1]
        X[:, ky, kx] = scrambled
        spatial = self._idct2d(X)                                    # (B, 32, 32)
        grid = spatial.unsqueeze(1)                                  # (B, 1, 32, 32)
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
        pooled = F.adaptive_avg_pool2d(tensor_rec, (GRID, GRID))     # (B, 12, 32, 32)
        S = pooled.mean(dim=1)                                       # (B, 32, 32)
        X = self._dct2d(S)                                           # (B, 32, 32)
        ky = self.coeff_idx[:, 0]
        kx = self.coeff_idx[:, 1]
        coeffs = X[:, ky, kx]                                        # (B, n_bits)
        unscrambled = coeffs * self.sign_mask
        return unscrambled                                           # logits, (B, n_bits)
```

### 4.4 Why the clean path is exactly invertible

1. `iDCT(X) → spatial 32×32 → tile to (h_dwt, w_dwt) → broadcast to 12 ch`.
2. Decode: `mean over 12 ch → adaptive_avg_pool2d to (32, 32) → DCT`.
3. The compositions `tile → mean over channels → adaptive_avg_pool2d` are the identity on the 32×32 spatial grid (block-broadcast then block-mean of a constant block returns the constant; channel-broadcast then channel-mean is identity).
4. `DCT(iDCT(X)) = X` exactly, since the orthonormal DCT-II's inverse is its transpose.
5. Reading off the same indices and multiplying by the sign-mask twice recovers the bipolar bits.

So `decode(encode(bits)) = bits` (up to fp32 round-off ≪ 1) — sign-perfect.

### 4.5 Diagnostic invertibility assertion (one-shot, step 0)

Add to `code/exp0_inn_train.py`, immediately after model construction:

```python
with torch.no_grad():
    _bits = torch.randint(0, 2, (2, args.secret_len), device=device).float()
    _car = model.adapter.encode(_bits, h_dwt=args.resolution // 2,
                                 w_dwt=args.resolution // 2)
    _rec = model.adapter.decode(_car)
    _ok = ((_rec.sign() + 1) / 2 == _bits).all().item()
    if not _ok:
        raise RuntimeError("PatchBitAdapter (DCT mid-band) clean-path "
                           "invertibility broken: sign mismatch")
    _err = (_rec - (_bits * 2.0 - 1.0) * model.adapter.sign_mask
             * model.adapter.sign_mask).abs().max().item()
    if _err > 1e-3:
        raise RuntimeError(
            f"PatchBitAdapter (DCT) clean-path numerical drift too large: "
            f"max |err| = {_err:.2e} > 1e-3"
        )
    print(f"[adapter] dct-midband invertibility check: PASS (max err {_err:.2e})")
```

(The `* sign_mask * sign_mask` cancellation in the error formula reduces to `(bits*2-1)`; written that way to make the relationship explicit and avoid a Builder typo.)

## 5. Logging spec

Add three diagnostic fields to the per-step `rec` dict:

```python
adapter_kind="dct_midband",
adapter_midband_lo=float(model.adapter.DCT_R_LO),
adapter_midband_hi=float(model.adapter.DCT_R_HI),
```

No other logging change. `metrics.jsonl` schema stays a strict superset of baseline.

## 6. Expected behaviour after the change

- **Forward cost:** two batched 32×32 matmuls per encode (idct = D^T @ X @ D) and two per decode (dct = D @ S @ D^T). Per call: 2 × B × 32 × 32 × 32 ≈ 1e6 multiply-adds, well under 0.01 % of one DKiS block. Negligible.
- **Carrier appearance:** the 32×32 spatial grid contains a sum of 896 mid-frequency 2D sinusoids with random ±1 weights — a textured, non-periodic mid-frequency pattern. After tile + iDWT, container_rgb's bit-carrier component looks like band-limited stochastic texture, with no periodic spike at the patch-grid frequency.
- **Compression robustness intuition:** mid-band DCT coefficients (frequencies 0.1 < r < 0.9 of Nyquist) sit in the part of JPEG's luminance quantization table where Q values are moderate (typical range 8–40 at QF=75) — they're attenuated but not zeroed out. JPEG-attack bit_acc should be at least as good as baseline (which already saturates at 0.999); the improvement headroom is on PSNR, not bit_acc.
- **`metrics.jsonl`** per-step keys = baseline keys ∪ `{adapter_kind, adapter_midband_lo, adapter_midband_hi}`.
- **`ckpt_latest.pt`** payload now includes new buffers `adapter.dct_d` (32×32 fp32 = 4 KiB), `adapter.dct_dt` (4 KiB), `adapter.coeff_idx` (n_bits×2 LongTensor = ~14 KiB), plus the smaller `sign_mask` (n_bits fp32 = ~3.5 KiB instead of 4 KiB). Negligible size delta.
- **Eval `metrics.csv`:** target `score >= 27.5` (≥ baseline + 0.25). Hard condition: `bit_acc_clean = 1.000` at every one of the 9 resolutions; structurally guaranteed by orthonormal DCT plus matched mean-pool decode.

## 7. Invariants Builder must preserve

1. **Locked flags unchanged:** `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.
2. **Locked infra unchanged:** zero edits under `infra/**`. `PatchBitAdapter.encode/decode` keep their public signatures.
3. **Hard condition:** `bit_acc_clean = 1.000` at every eval resolution. Structurally guaranteed by orthonormal DCT plus matched mean-pool decode; step-0 sign + 1e-3 numeric assertion in §4.5 guards implementation slips.
4. **Determinism:** the DCT matrix and coefficient enumeration are deterministic by construction; sign-mask uses a fixed seed (matching baseline). Two runs with the same RNG seed produce byte-identical containers.
5. **Mid-band annulus boundaries:** `DCT_R_LO=0.10`, `DCT_R_HI=0.90` are class-level constants. Builder must verify at construction that `_select_midband_coeffs` returns ≥ 896 pairs and raise `ValueError` immediately if not. Do not silently widen the annulus at runtime.
6. **Resume correctness:** the new buffers (`dct_d`, `dct_dt`, `coeff_idx`) are persistent and round-trip via `state_dict`. The `coeff_idx` buffer is a LongTensor — checkpoints must preserve dtype.
7. **Sign-mask length change:** baseline's `sign_mask` is length 1024; this design uses length `n_bits` (896). Builder must update the buffer registration accordingly (a length mismatch on resume from a checkpoint with the old length must raise, not silently truncate). Since this design starts from `baseline/` (a fresh-init run, not a resume from a baseline ckpt), there is no cross-design ckpt to migrate.
8. **No new CLI flags.** Annulus boundaries are class-level constants.

## 8. Edge cases

- **DCT scaling:** the orthonormal DCT-II convention gives `D @ D.T = I`. Any other convention (e.g., the `2*cos(...)` "unnormalized" form) would put a constant scale factor in the round-trip and the sign-mask would still recover the bit, but the carrier amplitude would be wrong by a factor — Builder must use the exact formula in §4.1.
- **Coefficient amplitude after iDCT:** with 896 ±1 coefficients spread over 1024 spatial positions, by Parseval the spatial RMS is `sqrt(896/1024) ≈ 0.935`, so the carrier magnitude is ~1 — same scale as baseline. No loss-weight retuning is required.
- **Eval at smaller resolutions (e.g. 64-px RGB → 32×32 DWT):** rep_h = rep_w = 1; tile is a no-op; decode pool is a no-op. DCT round-trip still applies and is sign-perfect.
- **Eval at larger resolutions (e.g. 1024-px RGB → 512×512 DWT):** rep_h = rep_w = 16. The 32×32 mid-frequency pattern is upsampled by 16× — meaning each 16×16 block is a constant equal to one spatial value. The decode `adaptive_avg_pool2d` to 32×32 averages each block back to that constant. DCT round-trip is then exact. Sign-perfect on clean.
- **fp16 / autocast:** the explicit `.to(dtype)` casts on `dct_d` / `dct_dt` keep matmuls inside autocast at fp16. Numeric error scales as O(32) × fp16 ulp ≈ 1e-3 — at the edge of the 1e-3 tolerance. The §4.5 numeric check is run in fp32 (outside autocast) so it never spuriously fires; the in-loop forward path uses fp16 like the rest of the network. Sign recovery is preserved because the per-coefficient logit magnitude is ~1.
- **Coefficient-count safety:** if a future change widens `n_bits` beyond what the annulus supports, `_select_midband_coeffs` raises immediately at construction time — not at training step 0. This makes the failure mode loud and fast.
