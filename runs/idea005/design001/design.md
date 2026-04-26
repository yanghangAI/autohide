**Design Description:** Replace the contiguous nearest-neighbor patch tile in `PatchBitAdapter` with a fixed deterministic balanced random pixel-permutation: each of the 1024 slots owns exactly 16 scattered DWT-spatial positions (at 256×256 / DWT 128×128) instead of one 4×4 contiguous block, and decode replaces `adaptive_avg_pool2d` with a per-slot scatter-mean over the same permutation. Zero-parameter, exactly invertible on clean.

**Parent:** baseline/

**Starting Point:** baseline/

---

## 1. Goal

Eliminate the visible 32×32 grid by removing spatial periodicity of the bit carrier while preserving per-bit ownership of an equal-area set of positions. This is the minimal-edit × scramble corner of the 2×2 design family. Expectation: container PSNR rises modestly (the bit energy no longer concentrates at piecewise-constant block boundaries that align to a visible 8-px grid at 256-px), while `bit_acc_clean = 1.000` is preserved exactly because the per-slot averaging still matches the per-slot embedding (orthogonal, balanced).

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

Builder may only touch the following files inside `code/` (mirroring `baseline/`):

- `code/inn_model.py`
- `code/exp0_inn_train.py` (only for the step-0 invertibility assert and one new diagnostic log key)

Builder must **not** touch any of:
- `code/losses.py`
- `infra/**` (locked)

## 4. Algorithmic spec — exact changes to `code/inn_model.py`

### 4.1 Add a permutation-builder near the top of the file (right after `_make_sign_mask`, before `class PatchBitAdapter`)

```python
def _make_balanced_permutation(num_slots: int = NUM_SLOTS,
                               num_positions: int,
                               seed: int = 0xD1FF5E) -> torch.Tensor:
    """Deterministic balanced assignment of `num_positions` flat positions to
    `num_slots` slots so that each slot owns exactly `num_positions // num_slots`
    distinct positions.

    Returns a LongTensor of shape (num_slots, positions_per_slot) where
    `out[s, j]` is the j-th flat position (in row-major order over the DWT
    spatial grid) owned by slot s.

    Construction (must be deterministic and reproducible across processes):
      1. p = num_positions // num_slots; require num_positions == p * num_slots.
      2. Build a length-`num_positions` LongTensor `assignments` equal to
         `arange(num_slots).repeat_interleave(p)` so each slot id appears
         exactly p times.
      3. Permute `assignments` with a torch.Generator(cpu).manual_seed(seed)
         using torch.randperm to drive a single Fisher-Yates shuffle.
      4. For each slot s, gather the positions where `assignments == s`
         (in ascending position-index order, deterministic) into row s.
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
    # Build (num_slots, p) — for each slot, the sorted list of its positions.
    out = torch.empty(num_slots, p, dtype=torch.long)
    counters = torch.zeros(num_slots, dtype=torch.long)
    for pos in range(num_positions):
        s = int(assignments[pos].item())
        out[s, int(counters[s].item())] = pos
        counters[s] += 1
    return out
```

The Python-loop fill is one-time at module construction and cheap (<10 ms for 16384 positions).

### 4.2 Replace the body of `PatchBitAdapter` with the permutation-based variant

Keep the class name, constructor signature `__init__(self, n_bits: int = 896)`, and method names `encode(bits, h_dwt, w_dwt)` and `decode(tensor_rec)` exactly as today. Internally:

- The adapter must support multiple DWT-spatial sizes since eval runs at 9 resolutions (DWT 16×16 up to 256×256). The permutation depends on `(h_dwt * w_dwt)` and on `num_slots = 1024`. Cache one permutation per `(h_dwt, w_dwt)` it has seen, lazily, in a `dict[tuple[int, int], LongTensor]` registered as a non-persistent buffer-holder (we cannot put a dict in `register_buffer`, so use a `_perm_cache: dict` plain attribute and persist the canonical 256×256 permutation as a buffer; non-canonical resolutions rebuild deterministically from the same seed at runtime).
- The seed must match the seed in `_make_balanced_permutation` (`0xD1FF5E`), so the permutation is reproducible across runs and processes.

Replacement class body:

```python
class PatchBitAdapter(nn.Module):
    """Bit ↔ tensor adapter via fixed balanced random pixel-permutation.

    Forward (encode): bits (B, n_bits) → bipolar ±1 → pad to 1024 slots →
    sign-mask scramble → scatter each slot's value to its `p` owned flat
    positions on the DWT spatial grid → broadcast across 12 channels.

    Reverse (decode): tensor (B, 12, h_dwt, w_dwt) → mean over 12 channels →
    flatten to (B, h_dwt*w_dwt) → for each slot s, average the values at
    its owned positions (via index_select + mean) → undo sign-mask → take
    first n_bits.
    """

    def __init__(self, n_bits: int = 896):
        super().__init__()
        if n_bits > NUM_SLOTS:
            raise ValueError(f"n_bits={n_bits} > {NUM_SLOTS} (1024 slots)")
        self.n_bits = n_bits
        sign = _make_sign_mask(NUM_SLOTS)
        self.register_buffer("sign_mask", sign, persistent=True)

        # Canonical-resolution permutation: 256×256 RGB → 128×128 DWT →
        # 16384 positions, 16 per slot. Persisted so resume after retraining
        # with a different shuffle would be detected as a checkpoint mismatch.
        canonical_positions = (256 // 2) * (256 // 2)
        canonical_perm = _make_balanced_permutation(
            num_slots=NUM_SLOTS,
            num_positions=canonical_positions,
            seed=0xD1FF5E,
        )
        self.register_buffer("perm_canonical", canonical_perm, persistent=True)
        # Runtime cache for non-canonical eval resolutions.
        self._perm_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _perm_for(self, h_dwt: int, w_dwt: int, device) -> torch.Tensor:
        """Return the (NUM_SLOTS, p) permutation tensor for this DWT size."""
        if h_dwt == 128 and w_dwt == 128:
            return self.perm_canonical.to(device=device)
        key = (int(h_dwt), int(w_dwt))
        cached = self._perm_cache.get(key)
        if cached is not None and cached.device == device:
            return cached
        # Rebuild deterministically; seed depends on size so different
        # resolutions get different shuffles (avoids accidental alignment).
        seed = 0xD1FF5E ^ (h_dwt * 1000003 + w_dwt)
        perm = _make_balanced_permutation(
            num_slots=NUM_SLOTS,
            num_positions=h_dwt * w_dwt,
            seed=seed,
        ).to(device=device)
        self._perm_cache[key] = perm
        return perm

    def encode(self, bits: torch.Tensor, h_dwt: int, w_dwt: int) -> torch.Tensor:
        B = bits.size(0)
        bipolar = bits * 2.0 - 1.0
        full = bits.new_zeros((B, NUM_SLOTS))
        full[:, : self.n_bits] = bipolar
        scrambled = full * self.sign_mask          # (B, 1024)
        perm = self._perm_for(h_dwt, w_dwt, bits.device)  # (1024, p)
        # Scatter: for slot s, write `scrambled[:, s]` into all p positions
        # listed in perm[s]. Implementation: gather slot ids per position.
        positions_to_slot = bits.new_empty(h_dwt * w_dwt, dtype=torch.long)
        # perm[s, j] is the flat position assigned to slot s.
        # Invert: position → slot.
        positions_to_slot.scatter_(0, perm.reshape(-1),
                                    torch.arange(NUM_SLOTS, device=bits.device)
                                    .repeat_interleave(perm.size(1)))
        # Build flat carrier (B, h*w) by gathering scrambled[:, slot_of_pos].
        flat = scrambled.index_select(1, positions_to_slot)  # (B, h*w)
        carrier = flat.view(B, 1, h_dwt, w_dwt)
        return carrier.expand(B, INN_CHANNELS, h_dwt, w_dwt).contiguous()

    def decode(self, tensor_rec: torch.Tensor) -> torch.Tensor:
        B, C, h_dwt, w_dwt = tensor_rec.shape
        avg_chan = tensor_rec.mean(dim=1)              # (B, h_dwt, w_dwt)
        flat = avg_chan.reshape(B, h_dwt * w_dwt)      # (B, h*w)
        perm = self._perm_for(h_dwt, w_dwt, tensor_rec.device)  # (1024, p)
        # Per-slot mean: gather positions, mean over p.
        # flat[:, perm] → (B, 1024, p)
        gathered = flat.index_select(1, perm.reshape(-1)).view(B, NUM_SLOTS, perm.size(1))
        per_slot = gathered.mean(dim=2)                # (B, 1024)
        unscrambled = per_slot * self.sign_mask
        return unscrambled[:, : self.n_bits]
```

The construction guarantees: each slot owns exactly `(h_dwt*w_dwt)/1024` positions; for 256-px input that's 16. Decode is the exact left-inverse of encode on the clean path because the per-slot averaging is over the same set of positions where the scrambled value was scattered, with no overlap between slots (balanced disjoint partition).

### 4.3 Diagnostic invertibility assertion (one-shot, step 0)

Add to `code/exp0_inn_train.py`, immediately after model construction (around the same place where the EMA was constructed in design004; here, right after the `print(f"[model] INNCodec n_params=...")` line). Skip if it already exists (idempotent).

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

The check passes by construction (each slot's encode value is broadcast to its 16 positions; decode mean recovers it exactly; sign-mask is its own inverse).

## 5. Logging spec

Add one diagnostic field to the per-step `rec` dict (around the existing logging block where new keys are appended):

```python
adapter_kind="balanced_perm",
```

This is a constant string per design and lets the aggregator distinguish runs. No other logging change. `metrics.jsonl` schema stays a strict superset of baseline.

## 6. Expected behaviour after the change

- **Forward cost:** the encode path replaces one `repeat_interleave` with one `index_select` over (h*w) positions; decode replaces `adaptive_avg_pool2d(.,(32,32))` with one `index_select`+`view`+`mean`. Both are O(h*w*B*12); negligible vs. the INN forward. Wallclock should be within ±2 % of baseline.
- **Container appearance:** the 8-px-pitch grid in container_rgb (visible at 256-px) is replaced by a high-frequency speckle pattern of the same total energy. The Fourier spectrum of the carrier becomes broadband (no spike at the patch-grid frequency).
- **`metrics.jsonl`** per-step keys = baseline keys ∪ `{adapter_kind}`.
- **`ckpt_latest.pt`** payload now includes the new buffer `adapter.perm_canonical` (LongTensor 1024×16) and the same `sign_mask` as baseline.
- **Eval `metrics.csv`:** target `score >= 27.5` (i.e. ≥ baseline + 0.25). Hard condition: `bit_acc_clean = 1.000` at every one of the 9 resolutions. Because each eval resolution rebuilds its own balanced permutation from the same deterministic seed schedule, the per-resolution clean path remains exactly invertible — the per-slot averaging is the exact left-inverse of the per-slot scatter at every (h_dwt, w_dwt).

## 7. Invariants Builder must preserve

1. **Locked flags unchanged:** `--batch_size 16`, `--secret_len 896`, `--resolution 256`, `--max_steps 30000`, `--total_steps 30000`.
2. **Locked infra unchanged:** zero edits under `infra/**`. The eval script's contract (calls `model.adapter.encode/decode` indirectly via `model.embed/extract`) is honoured by keeping `PatchBitAdapter`'s public method signatures (`encode(bits, h_dwt, w_dwt)`, `decode(tensor_rec)`) and class name.
3. **Hard condition:** `bit_acc_clean = 1.000` at every eval resolution. Structurally guaranteed by the disjoint-balanced partition: encode scatters one ±1 to 16 positions, decode means them — one is the exact inverse of the other up to floating-point. The step-0 assertion in §4.3 guards against implementation slips.
4. **Determinism:** the canonical permutation is a persistent buffer; non-canonical permutations are rebuilt from a fixed seed schedule (`0xD1FF5E ^ (h*1000003 + w)`), so two runs with the same seed produce byte-identical containers.
5. **Resume correctness:** the new `perm_canonical` buffer round-trips via `state_dict`. Re-loading a checkpoint trained with this design into a `PatchBitAdapter` constructor reuses the persisted buffer; `_perm_cache` is a runtime-only cache and is rebuilt deterministically from seeds.
6. **No new CLI flags.**

## 8. Edge cases

- **Eval at 32-px input (DWT 16×16, 256 positions):** `256 / 1024` is not an integer. The adapter must **never** be called at this size since the existing `_check_dims` already requires `H % 64 == 0`, so the smallest legal RGB input is 64×64 (DWT 32×32, 1024 positions, p=1). At p=1 each slot owns exactly one position — the balanced-permutation degenerates to a pure permutation, still exactly invertible. The eval suite's 9 resolutions (defined in locked `infra/exp0_inn_eval.py`) all satisfy `H % 64 == 0`, so this is safe.
- **Fp16 / autocast:** `index_select` and `mean` are autocast-safe; `LongTensor` indices are precision-agnostic. The decode mean is computed at the autocast dtype, identical to the baseline `adaptive_avg_pool2d` path.
- **`gather` vs `index_select`:** the spec uses `index_select` for clarity, but Builder may use `torch.gather` if it benchmarks faster — semantics are identical.
