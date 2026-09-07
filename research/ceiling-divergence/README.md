# ceiling-divergence — how far does Tsetlin.jl depart from the FPTM paper?

**Status: answered. The prediction held — the divergence is real but confined to 0.5% of clauses.**
Data-free; run `julia run.jl`, output in `results.txt`.

## Question

Three sources define the fuzzy clause's vote ceiling, and they do not all agree.

| Source | Ceiling |
|---|---|
| FPTM paper, arXiv:2508.08350 §2 | `n == 0 ? LF : min(n, LF)` |
| FuzzyPatternTM (Hnilov's reference impl) | `0 < n < LF ? n : LF` — **identical** |
| Tsetlin.jl (Hnilov's optimized rewrite) | `max(0, LF - misses)` — **flat `LF`** |

Two of three agree; the optimized rewrite is the outlier, and still is at upstream HEAD
(94e7828, 2026-08-21), so this is not a stale-copy artifact. The forms coincide when a clause holds
0 or at least `LF` literals and differ strictly in between. How much does that band matter?

## Pass/fail, stated before running

The gap is a training-time phenomenon that closes at convergence — a converged model should hold
almost no clauses in the divergence band. If instead many clauses sit below `LF` at convergence, the
two implementations are not training the same model and every published number needs re-reading.

## Setup

Hnilov's published 40-clause MNIST model, `L`=10, `LF`=5, 400 clause slots. No dataset needed: the
answer is entirely in the model's literal counts.

## Answer

**Prediction confirmed.**

```
empty (n = 0)          :    0    0.0%
divergence (0 < n < LF):    2    0.5%
at or above LF         :  398   99.5%

aggregate ceiling  paper/FuzzyPatternTM 1992   Tsetlin.jl flat 2000
                   flat over-votes by      8   (0.40% higher)
```

Two clause slots hold a single literal each; every other clause holds 12 or more. Under the paper
those two contribute at most 1 vote, under flat `LF` up to 5 — a 5x over-vote, on 0.5% of clauses,
worth 0.40% of total ceiling mass. With `T`=10 the class vote saturates long before either ceiling,
so the practical effect is smaller still.

**Flat `LF` is a safe fast path for a converged model.** The paper's ceiling matters during training
and for degenerate or heavily pruned clauses, not at the operating point.

## The unplanned finding: `L` is a growth gate, not a cap

`L`=10, yet clauses hold **12 to 54** literals, median ~18. Every clause in the model exceeds its
own literal cap, most of them by 2-5x.

The mechanism is visible in both implementations — FuzzyPatternTM line 256, Tsetlin.jl equivalently:

```julia
if length(literals[j]) + length(literals_inverted[j]) <= tm.L
    # ... then increment EVERY matching literal's automaton
```

The check runs *before* an increment pass that can push many automata across the include threshold
at once. So `L` does not bound clause size; it decides whether a clause is *allowed to grow this
round*. Equilibrium is set by where growth and forgetting balance, which lands far above `L`.

There is also a gap in the distribution: two clauses at 1 literal, then nothing until 12. Once a
clause starts growing it clears the gate's neighbourhood in a burst rather than creeping.

This matters beyond bookkeeping. `L` is documented as "max literals per clause" and reasoned about
as a capacity bound — including in the FPTM paper's hyperparameter guidance, where `LF <= L` is
offered as a rule of thumb. If `L` is really a growth gate, that relation does not mean what it
appears to.
