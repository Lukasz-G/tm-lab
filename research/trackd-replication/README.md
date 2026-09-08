# trackd-replication — do the Track D conclusions survive a second dataset?

**Status: direction replicates on all five variants; magnitudes do not.**
`julia --project=. run.jl [fashion|mnist] [epochs]`; output in `results-fashion.txt`.

## Why

Every Track D variant was tested on MNIST alone and all four lost, explained by one mechanism:
reducing a clause's tolerance costs it the redundant literals that make it robust. That is a
satisfying story resting on a single dataset at a single hyperparameter setting.

There is direct evidence that such stories are fragile here — the `mask-mining` interpretability
result looked clean on MNIST and collapsed on IMDb. So the variants were re-run on Fashion-MNIST,
which shares MNIST's format and shape but is harder.

Hyperparameters are held at the MNIST values deliberately. They are probably not optimal for
Fashion-MNIST, but the question is whether the **variants** behave the same relative to a common
baseline, not whether the baseline is well tuned. Retuning per dataset would reintroduce the
confound this is meant to remove.

## Result

Fashion-MNIST, 40 clauses/class, `T`=10, `S`=125, `L`=10, `LF`=5, 20 epochs, 3 seeds.

| variant | mean best | vs baseline | worse on | MNIST equivalent |
|---|---|---|---|---|
| baseline (published) | 0.8584 | — | — | — |
| proportional feedback | 0.8499 | **−0.0085** | 3/3 | −0.0042 |
| proportional-idle | 0.8519 | **−0.0064** | 3/3 | −0.0040 |
| confidence-weighted miss cost | 0.8565 | **−0.0019** | 3/3 | −0.0028 |
| anneal `LF` 5→1, `T` rescaled | 0.8545 | **−0.0039** | 3/3 | −0.0050 |
| `L` gates Type II too | 0.7969 | **−0.0615** | 3/3 | −0.1339 |

**Every variant loses, on every seed, on both datasets.** The sign replicates five times out of five.

**Magnitudes do not.** Proportional feedback is twice as damaging on Fashion-MNIST (−0.0085 against
−0.0042); capping Type II is less than half as damaging (−0.0615 against −0.1339). So effect sizes
are dataset-specific and should not be quoted as properties of FPTM.

**The mechanism signature replicates too**, which is the part that matters for the explanation rather
than the ranking. Clause sizes shrink for every variant that loses — median 14 at baseline against
13, 13, 12 and 11 — and the Type II cap reproduces its runaway-growth signature, maximum clause size
blowing from 34 to 151 as clauses that cannot learn to reject keep drawing reinforcement.

## What this does and does not buy

**Does:** the Track D claims are no longer single-dataset. The direction of every effect, and the
clause-size mechanism offered to explain it, hold on a second dataset with no retuning.

**Does not:** Fashion-MNIST is a weak replication. It is 28x28 grayscale images in the same file
format with the same class count — closer to a different sample of the same problem than to an
independent test. A genuinely independent check means CIFAR-10, or the noisy Amazon Sales set where
flat FPTM is known to beat GraphTM. Two image datasets of identical shape is better than one and is
not the same as two.

Also note the baseline here, 0.8584, is **not** a published Fashion-MNIST number — those use
convolutional booleanization and far more clauses. This is a like-for-like variant comparison on a
deliberately unmodified setup, not a benchmark attempt.
