# tmcore-differential — does TMCore's evaluator match the reference exactly?

**Status: PASS.** `julia --project=. run.jl`; output in `results.txt`.

## Question

TMCore stores clauses as packed 64-bit include masks and counts failed literals with a branch-free
kernel plus popcount. FuzzyPatternTM stores them as two lists of literal indices and walks them.
Same semantics on paper, entirely different mechanics — so agreement is a real check rather than a
tautology, and any disagreement localises to a specific clause and input.

This is the correctness gate for the evaluator. Unit tests confirm it matches its own definition;
this confirms the definition matches the reference implementation, on a real trained model.

## Setup

Hnilov's published 40-clause MNIST model, full 10,000-image test set. 400 clause slots × 10,000
inputs = **4,000,000 comparisons per ceiling policy**. Booleanization orientation is chosen by
measured accuracy (raw 0.9755 vs transposed 0.1395), so the comparison runs on inputs the model
actually understands rather than on noise where everything would trivially agree at zero.

## Results

```
imported 20 banks, 400 clause slots
import round-trips losslessly

LiteralCapped vs FuzzyPatternTM : 4000000 comparisons, 0 mismatches
FlatLF vs FuzzyPatternTM        : 20000 mismatches, from 2 clause slots in the divergence band
satisfied mask                  : 0 disagreements with the naive index recomputation
```

Three things are checked, and the middle one is the interesting one.

**Exactness.** `LiteralCapped` is the ceiling FuzzyPatternTM implements, and the two agree on every
one of four million evaluations. Not "close" — identical integers.

**The policy is real, and its effect is exactly the predicted size.** `FlatLF` is Tsetlin.jl's
semantics, so it must *not* agree everywhere; a silent match would mean the policy parameter never
reached the evaluator. It disagrees on 20,000 comparisons — which is precisely 2 clause slots ×
10,000 inputs, the 2 slots being exactly those holding between 1 and `LF-1` literals. The
divergence is confined to the band predicted by [`ceiling-divergence/`](../ceiling-divergence/) and
touches nothing else, on every input.

**The satisfied mask means what it claims.** Decoding the packed mask back to literal indices
reproduces the naive recomputation `{i ∈ literals : x[i]} ∪ {i ∈ inverted : ¬x[i]}` with zero
disagreements. The measurement hook is faithful, not approximately faithful.

**Import is lossless.** Packing index lists into masks and unpacking them returns the same sets, so
nothing is quietly dropped on the way in.

## Why this is the right gate

It fails loudly for the failure modes that matter: an off-by-one in bit indexing, a chunk-boundary
mistake at width 784 (which is not a multiple of 64, so the final chunk is 16 bits of padding), a
sign error in the miss kernel, or a ceiling policy that is accepted but ignored. None of those
would show up in a smoke test, and several would show up in training only as slightly worse
accuracy — the kind of bug that gets attributed to hyperparameters for a week.

It also means the reproduction claim is now grounded: whatever TMCore goes on to do with feedback,
its *evaluator* is the published one.
