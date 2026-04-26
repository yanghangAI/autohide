## design003
**Verdict:** APPROVED

**Automated gate:** PASS (`review-check-implementation`: structural OK; `check-scope` PASS against parent `baseline/`; `verify-claims` 4/4 snippets matched)

**Algorithm fidelity:**
- `_make_hadamard_orthonormal` at `code/inn_model.py:103-112` builds the Sylvester Hadamard via doubling and divides by `sqrt(n)`, matching design §4.1. Power-of-two guard present.
- `PatchBitAdapter.__init__` at `code/inn_model.py:128-137` registers `sign_mask`, `hadamard`, and the pre-transposed `hadamard_t`, all persistent — matching §4.2.
- `encode` at `code/inn_model.py:139-155` does bipolar → pad-to-1024 → sign-mask → matmul by H → reshape to (1, 32, 32) → `repeat_interleave` to DWT spatial → broadcast to 12 channels — faithful to §4.2.
- `decode` at `code/inn_model.py:157-165` does `adaptive_avg_pool2d((32,32))` → channel-mean → matmul by H^T → unscramble → slice n_bits. The order `pool` then `mean` differs trivially from the spec's `mean` then `pool`, but both are linear and commute, and the cross-channel mean of a broadcast-equal tensor is the identity, so the result is identical.
- Step-0 sign + 1e-3 numeric assertion at `code/exp0_inn_train.py:251-267` matches §4.4. Runtime log: `[adapter] hadamard invertibility check: PASS (max err 0.00e+00)`.
- Logging key `adapter_kind="hadamard"` at `code/exp0_inn_train.py:440`, matching §5.

**Training-signal sanity:**
- Orthonormal H means `H @ H.T = I` exactly (modulo fp32 ulp); the runtime check shows max err = 0. `bit_acc_clean = 1.000` is structurally guaranteed.
- The matmul is autocast-safe (`.to(dtype)` on the buffer); gradient flows through `@` normally.
- Mini-train ran to completion: PSNR climbing 10.98 → 13.19, bit_acc hovering 0.96–1.00, no NaN — healthy curve for the rapid 50/100 schedule.
- Optimizer/loss composition unchanged from baseline; only adapter swapped.

**Strongest objection:** The 1024×1024 Hadamard buffer adds ~8 MiB to the checkpoint (4 MiB for `hadamard` + 4 MiB for the redundant `hadamard_t`). `hadamard_t` is recoverable from `hadamard.t().contiguous()` at load time and is therefore strictly redundant. Persisting it does not affect correctness but is wasteful. Not a blocker for approval; a future refactor could derive `hadamard_t` lazily.

**Fixes required:** none.
