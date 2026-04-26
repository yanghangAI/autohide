# Builder Memory

Structured log of mistakes the Builder has made, kept so future invocations can skim and avoid repetition.

**Entry format (append new entries at the bottom):**

```
## <YYYY-MM-DD> — <one-line title>
**What I did:** ...
**Why it was wrong:** ...
**How to avoid:** ...
**Source:** <who caught it — Reviewer / scope_check / verify_claims / user>
```

Scripts auto-append entries for Builder on `scope_check` and `verify_claims` failures. Reviewer appends entries for Designer or Builder whenever it issues REJECTED.

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea001/design001, undeclared changes: README.md
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea001/design001, undeclared changes: README.md
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea001/design001, undeclared changes: __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design002
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea001/design002, undeclared changes: __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design003
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea001/design003, undeclared changes: __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea002/design001, undeclared changes: README.md
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — verify_claims failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea002/design001, claim not in NoiseCurriculum.__init__: [file not found] self._hard_chains = torch.nn.ModuleList([ _HardChain(scale=0.5, q=85), # eval ch…
**Why it was wrong:** Fenced code blocks in implementation_summary.md are verified against the actual files they reference. A snippet that doesn't exist in the claimed file means the summary misrepresents what was changed.
**How to avoid:** Copy the real changed lines directly from the file into the summary's code block, and cite the file path immediately above the block. Run `python scripts/cli.py verify-claims <design_dir>` locally first.
**Source:** verify_claims

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design001, undeclared changes: README.md, __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design002
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design002, undeclared changes: README.md, __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design003
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design003, undeclared changes: README.md, __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design001, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design002
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design002, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design003
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design003, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea004/design001, undeclared changes: README.md
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design002
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea004/design002, undeclared changes: README.md
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design001, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design002
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design002, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design003
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design003, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design001, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design002
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design002, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design003
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design003, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea003/design001, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea005/design001, undeclared changes: README.md
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design002
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea005/design002, undeclared changes: README.md
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design003
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea005/design003, undeclared changes: README.md
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design004
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea005/design004, undeclared changes: README.md
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea005/design001, undeclared changes: README.md
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea005/design001, undeclared changes: __pycache__/inn_model.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design003
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea005/design003, undeclared changes: __pycache__/inn_model.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design004
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea005/design004, undeclared changes: __pycache__/inn_model.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design002
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea005/design002, undeclared changes: __pycache__/inn_model.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea006/design001, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea006/design001, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design002
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea006/design002, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design003
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea006/design003, undeclared changes: __pycache__/exp0_inn_train.cpython-311.pyc, __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea006/design001, undeclared changes: __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design002
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea006/design002, undeclared changes: __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design003
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea006/design003, undeclared changes: __pycache__/inn_model.cpython-311.pyc, __pycache__/losses.cpython-311.pyc
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check

## 2026-04-26 — scope_check failed for design design001
**What I did:** At design /work/pi_nwycoff_umass_edu/hang/autohide/runs/idea007/design001, undeclared changes: README.md
**Why it was wrong:** Every code change must be declared in implementation_summary.md's **Files changed:** list, and files matching integrity.immutable_paths must stay byte-identical to baseline.
**How to avoid:** Before finalizing an implementation, list every modified file in implementation_summary.md; never modify files under immutable paths (e.g. infra/**). Run `python scripts/cli.py check-scope <design_dir>` locally and resolve failures first.
**Source:** scope_check
