**Files changed:**
code/inn_model.py
code/exp0_inn_train.py

**Changes:**
- `code/inn_model.py`: Added `_make_hadamard_orthonormal` helper and replaced the `PatchBitAdapter` body with the orthonormal Walsh-Hadamard spreading variant. Encode multiplies the sign-mask-scrambled bit vector by a 1024×1024 Hadamard matrix scaled by 1/sqrt(1024), reshapes to 32×32 and tiles. Decode mean-pools to 32×32 then multiplies by H^T to invert. The H and H^T tensors are registered as persistent buffers.
- `code/exp0_inn_train.py`: Inserted the step-0 invertibility assert with both sign-perfection and 1e-3 numeric tolerance after the `[model] INNCodec n_params=...` print, and added `adapter_kind="hadamard"` to the per-step `rec` dict.

In `code/inn_model.py`, the Hadamard builder:

```python
def _make_hadamard_orthonormal(n: int = NUM_SLOTS) -> torch.Tensor:
    """Sylvester-construction Hadamard matrix of size n×n, scaled to be
    orthonormal: H @ H.T == I exactly (for n a power of two)."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of two; got {n}")
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.size(0) < n:
        H = torch.cat([torch.cat([H,  H], dim=1),
                        torch.cat([H, -H], dim=1)], dim=0)
    return H / float(n) ** 0.5
```

In `code/inn_model.py`, the encode path:

```python
    def encode(self, bits: torch.Tensor, h_dwt: int, w_dwt: int) -> torch.Tensor:
        B = bits.size(0)
        bipolar = bits * 2.0 - 1.0
        full = bits.new_zeros((B, NUM_SLOTS))
        full[:, : self.n_bits] = bipolar
        scrambled = full * self.sign_mask                  # (B, 1024)
        spread = scrambled @ self.hadamard.to(scrambled.dtype)  # (B, 1024)
        grid = spread.view(B, 1, GRID, GRID)
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
            raise RuntimeError("PatchBitAdapter (Hadamard) clean-path "
                               "invertibility broken: sign mismatch")
        _err = (_rec - (_bits * 2.0 - 1.0)).abs().max().item()
        if _err > 1e-3:
            raise RuntimeError(
                f"PatchBitAdapter (Hadamard) clean-path numerical drift "
                f"too large: max |err| = {_err:.2e} > 1e-3"
            )
        print(f"[adapter] hadamard invertibility check: PASS (max err {_err:.2e})")
```

In `code/exp0_inn_train.py`, the logging key:

```python
                       adapter_kind="hadamard")
```
