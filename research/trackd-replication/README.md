# trackd-replication — do the Track D conclusions survive other datasets?

**Status: yes for direction, on all three datasets and all five variants — 15 of 15. No for
magnitude, which varies fivefold and tracks task difficulty.**
`julia --project=. run.jl [mnist|fashion|cifar] [epochs] [seeds]`; output in `results-*.txt`.

## Why

Every Track D variant was tested on MNIST alone and all four lost, explained by one mechanism:
reducing a clause's tolerance costs it the redundant literals that make it robust. A satisfying story
resting on one dataset.

There was direct evidence such stories are fragile here — `mask-mining` looked clean on MNIST and
collapsed on IMDb. So the variants were re-run on Fashion-MNIST and then CIFAR-10.

Hyperparameters are held at the MNIST values throughout, deliberately. They are not optimal for the
other two, but the question is whether the **variants** move the same way against a common baseline,
not whether the baseline is well tuned. Retuning per dataset would reintroduce the confound this is
meant to remove.

## Result

| variant | MNIST | Fashion-MNIST | CIFAR-10 | worse on |
|---|---|---|---|---|
| baseline accuracy | 0.9724 | 0.8584 | 0.3298 | — |
| proportional feedback | −0.0042 | −0.0085 | **−0.0228** | 11/11 seeds |
| proportional-idle | −0.0040 | −0.0064 | **−0.0204** | 11/11 |
| confidence-weighted miss cost | −0.0028 | −0.0019 | **−0.0060** | 11/11 |
| anneal `LF` 5→1, `T` rescaled | −0.0050 | −0.0039 | **−0.0141** | 11/11 |
| `L` gates Type II too | −0.1339 | −0.0615 | **−0.0831** | 11/11 |

**Fifteen of fifteen dataset-variant combinations lose, on every seed.** On CIFAR-10 the effects run
3x to 28x their standard error, so they are not noise even against a baseline near 0.33.

## Two things the third dataset added

**1. The damage scales with task difficulty.** Proportional feedback costs 0.4 points on MNIST, 0.9
on Fashion-MNIST and 2.3 on CIFAR-10 — a fivefold spread ordered exactly by how hard the task is
(0.97, 0.86, 0.33 baseline). That is consistent with the mechanism rather than merely compatible
with it: redundant literals are insurance against a noisy or ambiguous input, so removing them costs
little where the signal is clean and a great deal where it is not.

It also means the MNIST numbers were the *most flattering* case for these variants. Reporting only
those understated the damage by up to a factor of five.

**2. The runaway-growth signature is starker.** Capping Type II drives maximum clause size to **1,947
literals of a possible 2,048** on CIFAR-10, against 151 on Fashion-MNIST and 305 on MNIST. Clauses
that cannot learn to reject keep drawing reinforcement until they include nearly everything. The
mechanism is not a story fitted to MNIST; it gets louder on harder data.

The clause-shrinkage signature replicates too: median 13 at baseline against 13, 12, 11, 11 for the
losing variants, and the annealed arm shows the same best-versus-final collapse as before (0.3157
best, 0.2544 final) as `LF` falls.

## What this does and does not buy

**Does:** the Track D conclusions are no longer single-dataset, and CIFAR-10 is a genuine change of
regime — 32x32 colour photographs with real backgrounds, not another 28x28 grayscale sprite. Both
the direction and the proposed mechanism travel.

**Does not:** effect sizes are not properties of FPTM and should never be quoted as such. Only the
sign, and the ordering by difficulty, are supported.

## Caveats

The CIFAR-10 setup is deliberately modest — grayscale, a 2-bit fitted thermometer, 20,000 training
images, 40 clauses per class — chosen so six arms and five seeds stay affordable. Its 0.33 baseline
is **not** a CIFAR benchmark attempt and should not be compared with published convolutional TM
numbers. Colour and convolution were dropped because neither is what is under test here.

Still three image-classification datasets. A text or graph task would test something these do not,
and the Amazon Sales set — where flat FPTM is known to beat GraphTM — remains the interesting gap.
