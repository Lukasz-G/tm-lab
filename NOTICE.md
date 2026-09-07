# Third-party notices

tm-lab is MIT-licensed (see [LICENSE](LICENSE)). It derives from, and is designed to interoperate
with, the projects below. All are MIT, so the licenses are compatible — but compatibility is not
attribution, and attribution here is a **code** obligation, not a citation.

## The rule

Any source file that transfers code, memory layout, or a non-obvious algorithm from an upstream
project carries the upstream copyright notice **inline, in that file**, in addition to this file.
A file that merely implements something described in a paper does not. When in doubt, add it.

Name the upstream and the specific thing taken:

```julia
# Portions derived from Tsetlin.jl (packed include-mask clause evaluation, `check_clause`).
# Copyright (c) 2024-2026 Artem Hnilov. MIT License. See NOTICE.md.
```

## Upstreams

### Tsetlin.jl — MIT

- Copyright (c) 2024-2026 Artem Hnilov
- `github.com/BooBSD/Tsetlin.jl`
- **Role:** memory layout and inner loop. The packed TA include-mask, the bitwise-AND/popcount
  clause evaluator, and the chunked input bitvector are the parts expected to transfer at code
  granularity. Files inheriting these carry the notice inline.

### FuzzyPatternTM — MIT

- Copyright (c) 2024-2026 Artem Hnilov
- `github.com/BooBSD/FuzzyPatternTM`
- **Role:** reference semantics for the Fuzzy-Pattern TM. Read for meaning, not copied. Attribution
  expected to stay at the citation level unless that changes.

### GraphTsetlinMachine — MIT

- Copyright (c) 2024 Ole-Christoffer Granmo / CAIR
- `github.com/cair/GraphTsetlinMachine`
- **Role:** reference Graph TM (CUDA). Feedback mechanics and message-binding semantics read from
  `kernels.py`. No code transferred so far; the CUDA does not port to Julia literally. Revisit if
  that changes.

## Papers

Cited, not licensed:

- Fuzzy-Pattern Tsetlin Machine — arXiv:2508.08350
- Graph Tsetlin Machine — arXiv:2507.14874

## Open

Per-package `LICENSE` files are not present yet. They become required if any package under
`packages/` is submitted to the Julia General registry — a consequence of the deliverable-form
decision (README, open question 3), not a separate one.
