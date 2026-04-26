**Files changed:**
code/inn_model.py
code/exp0_inn_train.py

**Changes:**
- `code/inn_model.py`: Added `_make_dct_ii_orthonormal` and `_select_midband_coeffs` helpers and replaced the `PatchBitAdapter` body with a DCT-II mid-band spreading variant. Encode scatters the n_bits sign-mask-scrambled values into the (B, 32, 32) frequency grid at deterministic mid-band (k_y, k_x) indices, applies the 2D inverse DCT (= D^T X D), tiles to the DWT plane, and broadcasts to 12 channels. Decode mean-pools to 32×32, applies the forward 2D DCT (= D S D^T), reads off the same indices, and unscrambles. The sign-mask is now length n_bits.
- `code/exp0_inn_train.py`: Inserted the step-0 invertibility assert with sign-perfection and 1e-3 numeric tolerance right after the `[model] INNCodec n_params=...` print, and added `adapter_kind="dct_midband"`, `adapter_midband_lo`, `adapter_midband_hi` keys to the per-step `rec` dict.

In `code/inn_model.py`, the DCT builder:

```python
def _make_dct_ii_orthonormal(n: int) -> torch.Tensor:
    """Orthonormal 1D type-II DCT matrix D of shape (n, n)."""
    import math
    m_idx = torch.arange(n, dtype=torch.float32)
    k_idx = torch.arange(n, dtype=torch.float32)
    arg = (math.pi * (2.0 * m_idx[None, :] + 1.0) * k_idx[:, None]) / (2.0 * n)
    D = torch.cos(arg)
    scale = torch.full((n,), (2.0 / n) ** 0.5, dtype=torch.float32)
    scale[0] = (1.0 / n) ** 0.5
    D = D * scale[:, None]
    return D
```

In `code/inn_model.py`, the mid-band selector:

```python
def _select_midband_coeffs(grid: int = GRID, n_bits: int = 896,
                            r_lo: float = 0.20, r_hi: float = 0.85) -> torch.Tensor:
    """Pick exactly `n_bits` 2D-DCT coefficient indices (k_y, k_x) on a
    grid×grid frequency grid, restricted to a mid-band annulus."""
    coords = []
    for k_y in range(grid):
        for k_x in range(grid):
            if k_y == 0 and k_x == 0:
                continue
            r = ((k_y * k_y + k_x * k_x) ** 0.5) / ((2.0 * (grid - 1) ** 2) ** 0.5)
            if r < r_lo or r > r_hi:
                continue
            coords.append((r, k_y, k_x))
    coords.sort()
    if len(coords) < n_bits:
        raise ValueError(
            f"mid-band annulus has only {len(coords)} valid coefficients; "
            f"need {n_bits}. Loosen r_lo or r_hi."
        )
    coords = coords[:n_bits]
    out = torch.tensor([[ky, kx] for (_r, ky, kx) in coords], dtype=torch.long)
    return out
```

In `code/inn_model.py`, the encode/decode bodies:

```python
    def encode(self, bits: torch.Tensor, h_dwt: int, w_dwt: int) -> torch.Tensor:
        B = bits.size(0)
        bipolar = bits * 2.0 - 1.0                                  # (B, n_bits)
        scrambled = bipolar * self.sign_mask                         # (B, n_bits)
        X = bits.new_zeros((B, GRID, GRID))
        ky = self.coeff_idx[:, 0]
        kx = self.coeff_idx[:, 1]
        X[:, ky, kx] = scrambled
        spatial = self._idct2d(X)                                    # (B, 32, 32)
        grid = spatial.unsqueeze(1)                                  # (B, 1, 32, 32)
```

In `code/exp0_inn_train.py`, the step-0 invertibility check:

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

In `code/exp0_inn_train.py`, the new logging keys:

```python
                       adapter_kind="dct_midband",
                       adapter_midband_lo=float(model.adapter.DCT_R_LO),
                       adapter_midband_hi=float(model.adapter.DCT_R_HI))
```
