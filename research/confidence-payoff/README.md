# confidence-payoff — does the gradient pay for itself?

**Status: no. Weighting by confidence still loses, even with real confidence to weight by. The
accuracy branch of this family is closed.** `julia --project=. run.jl [dataset] [epochs] [seeds]`;
output in `results-mnist.txt`.

## The chain being tested

Confidence-weighted miss cost failed originally because a reference-trained FPTM has no confidence
gradient — every included automaton sits on the include threshold. [`eviction/`](../eviction/) fixed
that: partial freeze plus reset eviction produces a real gradient (7% to 94% of automata above the
threshold) for 0 to 0.4 accuracy points.

That is only worth having if something earns it back.

## Result

MNIST, 40 clauses/class, 20 epochs, 3 seeds.

| arm | accuracy | vs A | vs B |
|---|---|---|---|
| A reference | 0.9712 | — | −0.0006 |
| B gradient, uniform cost | 0.9718 | +0.0006 | — |
| C gradient, weighted, calibrated (thr 142) | 0.9691 | −0.0021 | **−0.0027** |
| D gradient, weighted, nominal (thr 191) | 0.9704 | −0.0009 | **−0.0014** |

**Neither weighted arm beats B.** The criterion was stated before running, and it says the gradient is
decorative: the original failure was not caused by the confidence being absent. Weighting a literal's
miss by how confident the model is about it is simply a bad rule, and now there is no missing
ingredient left to blame.

Note D — the nominal midpoint at 191 — is now a *meaningful* threshold rather than the silent no-op
it was in [`misscost/`](../misscost/), because states actually reach it. It still loses.

## What this settles, and what it does not

**Settles:** confidence-weighted evaluation is dead on its own terms, not for want of a gradient.
That is the fourth negative in this family and the one that closes it, because it removes the
explanation the previous three could hide behind.

**Does not settle:** the gradient's *interpretability* use is untouched by this. Ranking literals
within a clause by confidence needs only that the gradient exist, not that weighting improve
accuracy — and it is the obvious route out of the `imdb-readability` dead end, where a clause is a
precise, unreadable 3,780-literal conjunction with no way to say which literals matter most. That is
a separate experiment and this result does not speak to it.

## Caveat

MNIST only, 3 seeds. Given the pattern that effects amplify on harder data, C and D would likely lose
by more on Fashion-MNIST and CIFAR-10 rather than turn around — but that was not run, since the
result needed is "does it beat B", and it does not, on the easiest dataset where it had the best
chance.
