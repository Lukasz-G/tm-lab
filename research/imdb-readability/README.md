# imdb-readability — are the extracted cores readable by a human?

**Status: answered, and the answer is no — for a measurable reason that says exactly when the
method does work.** Run `prepare.py` then `run.jl`; output in `results.txt`.

## Question

[`mask-mining/`](../mask-mining/) found that a fuzzy clause decomposes into a strict conjunctive core
plus a tolerance tail, and that positive-clause cores are high-precision rules. But it ran on MNIST,
where a literal is a pixel: "readable" there meant low-entropy and checkable, not meaningful. Nobody
learns anything from pixel 412.

IMDb is the honest test. Literals are words and n-grams, so an extracted core either reads as a
sentiment rule or it does not.

## Setup

Published binary configuration from FuzzyPatternTM's `imdb_minimal.jl`: `T`=18, `S`=1000, `L`=64,
`LF`=64, 12,800 chi2-selected features over 4-grams, one clause per class per polarity, 30 epochs.

The data pipeline is ours (`prepare.py`) for two reasons. Upstream's script needs `keras` and, more
importantly, **never writes the vocabulary** — it emits bits and a label, which is enough to train
and impossible to read. Since mapping literal 4,412 back to `"one of the worst"` is the entire point
here, the vocabulary is the deliverable. `keras.datasets.imdb` only downloads two files and applies a
documented index shift, so that is reproduced directly and the pipeline needs numpy and scikit-learn.

**Accuracy check first: 0.9012, against the published 0.9015.** So this is the published model's
behaviour being read, not a weaker stand-in.

## Result

```
--- POSITIVE sentiment, for clause 1 ---
    3843 included literals, fired on 19771 of 25000 test docs
    core: 3780 literals, matches 950 docs, P(POSITIVE) = 0.979  (base rate 0.500)
```

All four clause slots look like this: **3,400–4,200 literals, with cores of 3,375–4,127.** The cores
are genuinely precise — 0.957 and 0.979 in the correct direction, 0.027 and 0.047 in the negative
direction — and completely unreadable. A 3,780-term conjunction is not a rule a human reads.

### Why: the core/tail split needs a fuzziness ratio, and this configuration has none

The clause holds 3,843 literals with `LF` = 64, so it fires when at most 63 of them fail — **1.7%
tolerance**. A literal must therefore be satisfied in essentially every firing almost by arithmetic,
and the 95% core threshold selects 98% of the clause. There is no tail to separate.

Contrast MNIST, where the decomposition was informative: `LF` = 5 against 23.5 literals is **21%
tolerance**, and the core came out at 11 of 23.5.

So the operative quantity is **`LF` / included literals**, and it is a usable diagnostic rather than
a post-hoc excuse: compute it before expecting a decomposition. Below a few percent the clause is
already a near-strict conjunction wearing fuzzy clothing, and core extraction returns the clause back
to you.

This is also `L` as a growth gate at its most extreme. `L` = 64 and clauses hold ~3,800 literals —
**58x over**, against 5x on MNIST.

### The unplanned finding: the model is a pure blacklist

**Not one core requires any term to be present.** Across all four clause slots, every core literal is
a *negated* one — the model classifies entirely by which n-grams are missing.

That is mechanically unsurprising with 12,800 sparse features: any given n-gram is absent from nearly
every document, so negated literals are cheap to satisfy and positive ones are not. The learner takes
the cheap route. But it is worth stating plainly, because "one clause per class at 90% accuracy"
naturally reads as *the clause has learned what a positive review looks like*, and it has not. It has
learned a list of things a positive review does not contain.

The individual terms are sensible — the positive-sentiment core requires the absence of `"0 out of
10"`, `"1 star"`, `"1 out of 10"`, `"2 out of"` — so the *features* carry meaning even though the
*rule* does not.

## What this settles

The MNIST interpretability result does not transfer, and the reason is measurable rather than
mysterious. Core extraction repairs readability only when the clause has real tolerance to spend;
in the published IMDb configuration it has 1.7%, and the method returns a 3,780-literal conjunction.

The honest summary across both experiments: **FPTM clauses are recoverable as rules when `LF` is a
large fraction of the clause size, and not otherwise.** The published IMDb model — the one the paper
highlights for its 50x clause reduction and 50 KB footprint — sits firmly in the "not otherwise"
regime, and is additionally a pure exclusion model, which is a further step away from the kind of
rule the TM family is chosen for.

## Caveats and what would move it

One configuration. A model trained at high `LF`/`n` on the same data — fewer selected features, or a
harder `L` — might decompose readably and would be the natural follow-up; the ratio makes that a
testable prediction rather than a hope. The 95% core threshold was not swept, though at 1.7%
tolerance no threshold in a sensible range would change the conclusion.
