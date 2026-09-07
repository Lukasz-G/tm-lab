# aliasing — bit- versus symbol-granular fuzziness

**Status: answered.** Arithmetic only, no training. Run `julia aliasing.jl`; output in
`results.txt`.

## Question

FPTM's fuzzy clause lets a vote survive some number of failed literals. Over thermometer or
bag-of-words features a partial match is a meaningful sub-pattern. Over a sparse distributed code —
a symbol is `H` bits set out of `D` — missing a bit may instead produce a *corrupted symbol* that
aliases to every other symbol sharing the remaining bits. If so, bit-granular fuzziness is
semantically incoherent and fuzziness would have to be redefined at symbol granularity, with `LF`
counting whole-symbol misses.

## Pass/fail, stated before running

Bit-granular matching degrades toward chance as vocabulary `V` grows at fixed `H`, while
symbol-granular tracks strict.

## Model

`D` hypervector size, `H` bits set per symbol, `V` vocabulary, `m` symbols OR-ed into one node's
vector, `k` symbols conjoined by a clause (so `k*H` positive literals).

Assumptions, stated so they can be attacked:

- **A1** symbol codes are independent uniform `H`-subsets of `[D]`.
- **A2** a node's feature vector is the bitwise OR of the codes present.
- **A3** bits are independent in the union statistics. The pairwise overlap calculation is exact
  (hypergeometric); only the superposition tables lean on A3.

The headline metric is the **alias set size** `N(f)`: the expected number of *other* symbols in the
vocabulary that, on their own, drive a clause targeting symbol `s` to within `f` of a full vote.
`N(0)` counts exact code collisions and should be ~0 in any sane configuration.

## Answer

**The mechanism is real and the diagnosis was right. The prescribed fix is the wrong lever, and it
is not even the conservative choice.**

### 1. The inflation is real, large, and worst exactly where codes are sparsest

Allowing one missing bit multiplies the false-positive rate by a factor with a closed form:

```
inflation = 1 + n(1-p)/p        n = k*H required bits,  p = P(a bit is set)
```

Measured across the grid: **27x to 763x**. Sparse codes mean small `p`, so `(1-p)/p` is large. The
intuition that distributed codes punish partial matching is correct, and this is why.

### 2. But absolute rate is what decides, and `H` controls it outright

Alias set size `N(1)`, at `V` = 100:

| | H=1 | H=2 | H=4 | H=8 |
|---|---|---|---|---|
| D=32 | 99.0 | **12.2** | 0.31 | 1.8e-03 |
| D=256 | 99.0 | **1.54** | 5.7e-04 | 4.8e-10 |
| D=1024 | 99.0 | **0.39** | 8.9e-06 | 2.8e-14 |

At `H`=1 a clause fires on every symbol in the vocabulary — fuzziness is meaningless by
construction. At `H`=2, `D`=32 — the example in CLAUDE.md §5 — **12 other symbols each trigger a
near-full vote**. That configuration is incoherent, exactly as suspected. At `H`>=4 the alias set is
below 10^-3 and the concern evaporates. The danger zone is `H` <= 2 and it ends abruptly.

### 3. The stated pass/fail has no single answer — it has an `H` threshold

Alias growth with vocabulary at `D`=256:

| H | V=10 | V=100 | V=1000 | V=10000 |
|---|---|---|---|---|
| 2 | 0.14 | 1.54 | 15.6 | **156** |
| 4 | 5.2e-05 | 5.7e-04 | 5.8e-03 | 0.058 |
| 8 | 4.4e-11 | 4.8e-10 | 4.8e-09 | 4.8e-08 |

At `H`=2 the criterion **passes**: linear degradation, 0.14 to 156. At `H`=4 it **fails** — 0.058 at
ten thousand symbols. At `H`=8 it fails by twenty orders of magnitude. The hypothesis is confirmed
in one regime and refuted in the other, and `H` is the switch.

### 4. The surprise: symbol granularity is not safer, it is coarser

False-positive rate, `k`=3 symbols per clause, `D`=256:

| H | m | strict | bit LF=1 | bit LF=2 | **sym LF=1** |
|---|---|---|---|---|---|
| 4 | 16 | 1.5e-08 | 6.4e-07 | 1.3e-05 | **1.8e-05** |
| 8 | 4 | 6.9e-23 | 1.2e-20 | 1.0e-18 | **5.0e-15** |
| 8 | 16 | 2.5e-10 | 9.5e-09 | 1.7e-07 | **1.2e-06** |

Forgiving one *whole symbol* is a far bigger concession than forgiving one bit: it turns a 3-symbol
clause into a 2-symbol clause. At `H` >= 4, symbol-`LF`=1 admits **more** false positives than
bit-`LF`=2. So the §5 mitigation does not buy correctness.

What symbol granularity does buy is **interpretability**. "Matched 2 of 3 symbols" is a genuine
m-of-n rule over meaningful units; "missed 1 of 24 bits" is not a statement about anything. That is
a real benefit and it belongs to the measurement track, not to a correctness argument.

## Consequences

- **Track A is unblocked, and does not need a granularity parameter for correctness.** Bit-granular
  is safe at `H` >= 4 and is the cheaper implementation: the existing popcount inner loop, bit-packing
  intact, no symbol-boundary bookkeeping.
- **Add a construction-time guard instead.** Given `(D, H, V)`, compute `N(1)` and refuse or warn on
  fuzzy semantics when it exceeds a small threshold. Five lines, catches the incoherent regime
  before any training happens, and turns a silent failure into a loud one.
- **Symbol granularity becomes an optional evaluator mode motivated by interpretability**, scheduled
  with the measurement and graph tracks rather than as a prerequisite.

## The one caveat that could overturn this

**A1 fails for message symbols.** GraphTM binds messages by cyclic shift — `(bit + edge_type) %
MESSAGE_SIZE` — so message codes are permutations of each other rather than independent draws, and
they share a bit space across clauses *and* edge types. Overlaps there are structurally higher than
hypergeometric, and this analysis is optimistic for them.

This result therefore covers **node-property symbols**. The message-symbol space needs its own
calculation over the permutation structure, and that is the thing a toy would actually be worth
spending on — not the node case, which the arithmetic has settled.
