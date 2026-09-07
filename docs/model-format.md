# TMCore model format, version 2

A container for a trained Fuzzy-Pattern Tsetlin Machine. Deliberately boring: a fixed binary header
followed by raw packed arrays, little-endian throughout, no self-describing schema and no
language-specific serialization.

**Why not just serialize.** Julia's `Serialization` — which both upstream FPTM implementations use —
does not survive a Julia upgrade or a struct rename. Upstream renamed its central struct from
`TATeam` to `TMClauses` in 2026, which silently invalidates every model saved before it. A format
whose readers break when its *writer* is refactored is not a format.

**Why now, before it is needed.** The value of a model format is entirely in other implementations
adopting it, and "the Julia format with a Python bridge" does not get adopted. The cost of deciding
late is not the decision, it is accretion: whatever serialization exists when the question is
finally asked is what ends up standardised. Writing something a stranger can read in fifty lines
costs nothing today.

A reference reader lives in [`tools/read_tmcore.py`](../tools/read_tmcore.py) — pure Python, no
dependencies beyond the standard library, and it is checked against the Julia writer on a real
trained model rather than on a fixture.

## Conventions

- All integers little-endian.
- All offsets and sizes in bytes.
- Arrays are written **clause by clause**: for each clause, all of its values contiguously. For a
  `nchunks × nclauses` matrix this is Julia's native column-major order, and in NumPy it reads back
  as `reshape(nclauses, nchunks)` in C order.
- Literal position `i` (1-based) lives at bit `(i-1) mod 64` of chunk `(i-1) div 64`. Bits beyond
  `width` in the final chunk are always zero.

## Header

| Offset | Size | Type | Field |
|---|---|---|---|
| 0 | 8 | bytes | magic, ASCII `TMCORE\0\0` |
| 8 | 4 | uint32 | `version` (currently 2) |
| 12 | 4 | uint32 | `header_size` — byte offset where the payload begins |
| 16 | 4 | uint32 | `width` — input bits |
| 20 | 4 | uint32 | `nchunks` — `ceil(width / 64)` |
| 24 | 4 | uint32 | `nclasses` |
| 28 | 4 | uint32 | `clauses_per_polarity` |
| 32 | 4 | uint32 | `T` |
| 36 | 4 | uint32 | `S` |
| 40 | 4 | uint32 | `L` |
| 44 | 4 | uint32 | `LF` |
| 48 | 4 | uint32 | `include_limit` |
| 52 | 4 | uint32 | `state_min` |
| 56 | 4 | uint32 | `state_max` |
| 60 | 1 | uint8 | `ceiling_policy`: 0 = literal-capped, 1 = flat `LF` |
| 61 | 1 | uint8 | `budget_policy`: 0 = growth gate, 1 = hard cap |
| 62 | 1 | uint8 | `state_bytes`: 0 = automata omitted, 1 = uint8, 2 = uint16 |
| 63 | 1 | uint8 | `class_type`: 0 = int64, 1 = utf8 string, 2 = bool |
| 64 | 8 | uint64 | `payload_bytes` — length of the payload, for truncation detection |
| 72 | 4 | uint32 | `class_block_bytes` |
| 76 | 1 | uint8 | `feedback_policy`: 0 = threshold, 1 = proportional *(added in v2)* |
| 77 | 3 | bytes | reserved, must be zero; readers must reject a non-zero value |
| 80 | … | | class label block |

`header_size` = 80 + `class_block_bytes`, and the payload begins there.

`feedback_policy` affects training only and never inference, so a v1 model's predictions are
unambiguous without it. It is recorded because a checkpoint reloaded to continue training would
otherwise switch feedback rules silently.

### Class label block

- `class_type` 0 (int64) or 2 (bool): `nclasses` little-endian int64 values. For bool, 0 and 1.
- `class_type` 1 (string): `nclasses` records of `uint32 length` followed by that many UTF-8 bytes.

Classes are stored in the model's own order, which is ascending, and prediction returns the label at
the winning index.

## Payload

For each class `c` in order, then for each polarity in the order **positive, negative**:

| Size | Type | Field |
|---|---|---|
| `nchunks × clauses_per_polarity × 8` | uint64 | `included` |
| `nchunks × clauses_per_polarity × 8` | uint64 | `included_inv` |
| `clauses_per_polarity × 4` | int32 | `count` — included literals per clause |
| `width × clauses_per_polarity × state_bytes` | uint | `state`, omitted when `state_bytes` = 0 |
| `width × clauses_per_polarity × state_bytes` | uint | `state_inv`, omitted likewise |

`count` is stored rather than derived because it feeds the literal-capped ceiling on every
evaluation. It counts **literals, not positions**: a clause including both `x` and `¬x` at one
position counts two. A reader may recompute it as
`popcount(included) + popcount(included_inv)` and should verify agreement.

## Evaluating a clause from this file alone

Everything an independent implementation needs is here, so the format is checkable without trusting
the writer:

```
misses(clause j, input x) = sum over chunks n of
    popcount( ((included[n,j] XOR included_inv[n,j]) AND x[n]) XOR included[n,j] )

ceiling = literal-capped ? (count[j] == 0 ? LF : min(count[j], LF)) : LF
vote    = max(0, ceiling - misses)
```

Class score is the summed vote of its positive clauses minus that of its negative clauses, and the
prediction is the highest-scoring class, ties going to the lowest index.

## Versioning

`version` is bumped for any change that an existing reader would misparse. Readers must reject a
version they do not know rather than guess. Fields are never repurposed; the header grows only at
the end, and `header_size` means an older reader can still find the payload.

**Changes**

- **v2** — added `feedback_policy` at offset 76 and three reserved bytes, moving the class label
  block from 76 to 80. A v1 reader misparses a v2 file, which is exactly why the version is bumped
  rather than the byte quietly appended.
- **v1** — initial.
