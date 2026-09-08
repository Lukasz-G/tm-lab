# literal-ranking — can a confidence gradient make an unreadable clause readable?

**Status: no. The gradient saturates on IMDb, and the frequency baseline is no better. The
interpretability dead end stands.** `julia --project=. run.jl [epochs]`; output in `results.txt`.

## The idea

`imdb-readability` ended in a dead end: the published one-clause-per-class IMDb model is precise and
its rule is a ~3,800-literal conjunction. Worse, there was no way even to *rank* those literals —
every included automaton sat on the include threshold, so all of them looked equally important.

`eviction/` made a gradient available for 0 to 0.4 accuracy points. Weighting the vote by it does not
pay, but ranking needs only that it exist. So: rank each clause's literals by automaton state,
truncate to the top N, and see whether a short rule keeps the full rule's precision.

Baseline to beat: ranking by **satisfaction frequency**, which needs no gradient at all and is what
anyone would try first.

## Result

Accuracy 0.8985 (against 0.9012 without reset and a published 0.9015) — the expected small cost.

**But the gradient did not form.** Included automata: median **255**, max 255, 99.9% above the
threshold. Fully saturated, not spread. So "ranking by confidence" is really ranking by tie-break
order, which is why the top 12 comes out alphabetical: `"0"`, `"0 10"`, `"0 out of"`, `"0 out of
10"`, `"1"`, …

```
--- POSITIVE sentiment, for clause 1: 3517 literals, fires 19499/25000 ---
    full rule: matches 0 docs
    top-N   ranked by CONFIDENCE      ranked by FREQUENCY
    5       matches 23137  P 0.518    matches 24964  P 0.501
    20      matches 21510  P 0.530    matches 24911  P 0.502
    200     matches  1490  P 0.852    matches 23602  P 0.525
```

**Frequency ranking is useless too** — every truncation sits at P ≈ 0.50, chance. The most-satisfied
literals are the ones satisfied in nearly every document, i.e. rare n-grams absent from everything,
which carry no discriminative information at all.

Note also that the full 3,517-literal conjunction matches **zero** test documents. The 95%-core in
`imdb-readability` matched 950; requiring every literal is stricter than any document satisfies.

## Why the gradient saturated, which is the transferable part

Reset eviction produced a real spread on MNIST — median 146, 94% above the threshold — and none here.
The difference is **firing rate**. These IMDb clauses fire on 19,499 of 25,000 documents, 78%, so
included literals are reinforced almost every example and pin at `state_max`. MNIST clauses fire on
roughly 9% of inputs, leaving erosion enough room to hold states in the middle of the band.

So reset eviction yields a usable gradient only when clauses fire *infrequently* enough that
reinforcement does not saturate them. That is a condition worth stating before anyone relies on it,
and it is not something the MNIST result alone would have revealed.

## What is still open

Neither ranking signal here is discriminative, and that is the actual flaw rather than bad luck.
A literal's automaton state and its satisfaction frequency both measure *how often it holds*, not
*how much it separates the classes*. A discriminative score — satisfaction rate on in-class documents
against out-of-class ones — needs no automaton gradient and is the obvious next thing to try. It was
not tried here.

So this closes the confidence route to readability, not readability itself.
