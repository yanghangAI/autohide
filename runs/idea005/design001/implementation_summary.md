**Files changed:**
code/inn_model.py
code/exp0_inn_train.py

**Changes:**
- `code/inn_model.py`: Added `_make_balanced_permutation` helper and replaced the `PatchBitAdapter` body with a balanced random pixel-permutation variant. Each of the 1024 slots owns exactly `(h_dwt*w_dwt)/1024` scattered DWT positions; encode scatters slot values to those positions and decode means them back, preserving sign-perfect invertibility on the clean path.
- `code/exp0_inn_train.py`: Inserted the step-0 adapter-only invertibility assertion right after the `[model] INNCodec n_params=...` print, and added `adapter_kind="balanced_perm"` to the per-step `rec` dict for diagnostic logging.

In `code/inn_model.py`, the new helper:

```python
def _make_balanced_permutation(num_positions: int,
                               num_slots: int = NUM_SLOTS,
                               seed: int = 0xD1FF5E) -> torch.Tensor:
    """Deterministic balanced assignment of `num_positions` flat positions to
    `num_slots` slots so that each slot owns exactly `num_positions // num_slots`
    distinct positions.

    Returns a LongTensor of shape (num_slots, positions_per_slot) where
    `out[s, j]` is the j-th flat position (in row-major order over the DWT
    spatial grid) owned by slot s.
    """
    if num_positions % num_slots != 0:
        raise ValueError(
            f"num_positions={num_positions} must be a multiple of "
            f"num_slots={num_slots}"
        )
    p = num_positions // num_slots
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(num_positions, generator=g)
    assignments = torch.arange(num_slots).repeat_interleave(p)[perm]
    out = torch.empty(num_slots, p, dtype=torch.long)
    counters = torch.zeros(num_slots, dtype=torch.long)
    for pos in range(num_positions):
        s = int(assignments[pos].item())
        out[s, int(counters[s].item())] = pos
        counters[s] += 1
    return out
```

In `code/inn_model.py`, the new encode/decode body:

```python
    def encode(self, bits: torch.Tensor, h_dwt: int, w_dwt: int) -> torch.Tensor:
        B = bits.size(0)
        bipolar = bits * 2.0 - 1.0
        full = bits.new_zeros((B, NUM_SLOTS))
        full[:, : self.n_bits] = bipolar
        scrambled = full * self.sign_mask          # (B, 1024)
        perm = self._perm_for(h_dwt, w_dwt, bits.device)  # (1024, p)
        # Invert: position → slot.
        positions_to_slot = torch.empty(h_dwt * w_dwt, dtype=torch.long,
                                         device=bits.device)
        positions_to_slot.scatter_(0, perm.reshape(-1),
                                    torch.arange(NUM_SLOTS, device=bits.device)
                                    .repeat_interleave(perm.size(1)))
        # Build flat carrier (B, h*w) by gathering scrambled[:, slot_of_pos].
        flat = scrambled.index_select(1, positions_to_slot)  # (B, h*w)
        carrier = flat.view(B, 1, h_dwt, w_dwt)
        return carrier.expand(B, INN_CHANNELS, h_dwt, w_dwt).contiguous()
```

In `code/exp0_inn_train.py`, the step-0 invertibility check:

```python
    # Adapter-only invertibility check: encode(bits)→decode round-trip must be
    # exact (sign-perfect) regardless of INN training state.
    with torch.no_grad():
        _bits = torch.randint(0, 2, (2, args.secret_len), device=device).float()
        _car = model.adapter.encode(_bits, h_dwt=args.resolution // 2,
                                     w_dwt=args.resolution // 2)
        _rec = model.adapter.decode(_car)  # signs of (_rec) should match (_bits*2-1)
        _ok = ((_rec.sign() + 1) / 2 == _bits).all().item()
        if not _ok:
            raise RuntimeError("PatchBitAdapter clean-path invertibility broken")
        print(f"[adapter] balanced-permutation invertibility check: PASS")
```

In `code/exp0_inn_train.py`, the new logging key:

```python
                       adapter_kind="balanced_perm")
```
