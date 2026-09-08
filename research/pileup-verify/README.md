# pileup-verify — is the pile-up real in Hnilov's code, and is `L` the cause?

**Status: yes to both, confirmed in FuzzyPatternTM itself.**
`julia --project=. run.jl [epochs]`; output in `results.txt`.

## Why this was needed

The claim that a trained FPTM has no confidence gradient — every included automaton on the include
threshold — was measured only in TMCore. TMCore's feedback rules are verified identical to
FuzzyPatternTM's state-for-state, so the dynamics ought to match, but "ought to" was doing real work
there. Hnilov's published model is compiled, with automata discarded, so it could not be read
directly. The fix is to train with his code and read its automata.

## Result

MNIST, 20,000 training images, 40 clauses/class, `T`=10, `S`=125, `LF`=5, 5 epochs.

| implementation | `L` | accuracy | literals (median) | included state min/med/max | above threshold |
|---|---|---|---|---|---|
| FuzzyPatternTM | 10 | 0.9536 | 16 | 128 / **128** / 133 | **8.1%** |
| TMCore | 10 | 0.9517 | 15 | 128 / **128** / 136 | **7.7%** |
| FuzzyPatternTM | 100000 | 0.3114 | 542 | 128 / 152 / 255 | **97.0%** |
| TMCore | 100000 | 0.3422 | 523 | 128 / 158 / 255 | **97.0%** |

**The pile-up is real in the reference implementation.** At a published-style `L`, FuzzyPatternTM's
included automata sit at a median of exactly 128 with a maximum of 133 — barely off the threshold —
and only 8.1% are above it. This is a property of FPTM as published, not of this reimplementation.

**`L` is the cause.** Open the gate and it vanishes: 97.0% above the threshold, median 152, maximum
saturated at 255. Both implementations agree to within a percentage point on every measurement,
which incidentally cross-validates TMCore against Hnilov's code on a quantity no test covered.

**Removing `L` is not an option.** Accuracy collapses from 0.95 to 0.31 and clauses balloon to ~530
literals. That is exactly why [`eviction/`](../eviction/) matters: reset eviction decouples the
gradient from the gate rather than removing the gate.

## Scope

`L` is not part of Granmo's 2018 Tsetlin Machine; it is a later addition, present as
`max_included_literals` in the tmu lineage and `MAX_INCLUDED_LITERALS` in GraphTM's CUDA. So this is
a consequence of a knob added after the original design, and the classical TM presumably does
develop a gradient. That has not been measured here.

One dataset, 5 epochs, one clause count.
