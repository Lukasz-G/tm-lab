# booleanization — how much accuracy lives in the encoder?

**Status: answered for threshold count. The encoder is worth about +0.002 on MNIST — and more than
half of what a naive comparison would credit to it is a hyperparameter side-effect.**
`julia --project=. run.jl [epochs]`; output in `results.txt`.

## Question

The design notes claim much of the reported TM gain sits in booleanization rather than in the
machine. FuzzyPatternTM's own MNIST files disagree with each other about it: the training example
encodes at four thresholds, `booleanize(x, 0, 0.25, 0.5, 0.75)`, and the inference benchmark at one,
`booleanize(x, 0.25)`. Same machine, same hyperparameters, 4x the input width, and nobody reports
what the difference is worth.

## The confound, which turns out to be larger than the effect

`s` — the number of automata Type Ib erodes per event — is derived as `width / S`. Widening the
input from 784 to 3,136 bits therefore **quadruples the forgetting rate** unless `S` is scaled with
it. A naive 1-bit versus 4-bit comparison changes two things and credits the encoder with both.

So `S` is scaled to hold `s` = 6 across arms, and one control deliberately does not.

## Result

MNIST, 40 clauses/class, `T`=10, `L`=10, `LF`=5, 20 epochs, 3 seeds.

| encoder | width | S | s | mean best | vs 1-bit |
|---|---|---|---|---|---|
| fixed 1 bit (0.25) | 784 | 125 | 6 | 0.9712 | — |
| fixed 4 bit (0,.25,.5,.75) | 3136 | 500 | 6 | 0.9734 | +0.0021 |
| thermometer 2 bit (quantile) | 1568 | 250 | 6 | **0.9743** | **+0.0031** |
| thermometer 4 bit (quantile) | 3136 | 500 | 6 | 0.9727 | +0.0014 |
| fixed 4 bit, **S not scaled** | 3136 | 125 | 25 | 0.9759 | +0.0046 |

Decomposing the naive comparison:

```
encoder alone (1 -> 4 fixed bits, s held) : +0.0021
forgetting rate alone (s 6 -> 25)         : +0.0025
                                            -------
what a naive 1-bit vs 4-bit test reports  : +0.0046
```

**More than half of the apparent benefit of richer booleanization is not the booleanization.** It is
`s` moving, because `S` was left at a value tuned for a narrower input. Anyone comparing encoders
without holding `s` fixed is measuring a mixture and attributing it to the wrong component.

## Three things this says

**1. On MNIST, the encoder is worth little.** +0.002 for four bits per pixel instead of one. Real —
it is consistent across seeds — but small, and it does not support "much of the gain lives here" for
this dataset and this intervention.

**2. Fitted thresholds do not beat assumed ones**, at least here: the quantile thermometer at 4 bits
is 0.0007 *behind* upstream's hardcoded quartiles. MNIST pixels are mostly background, so
data-fitted quantiles cluster near zero and buy nothing that `0, 0.25, 0.5, 0.75` did not already
have. The library's value is not that fitting wins; it is that the choice is explicit, named, and
cheap to vary.

**3. More bits is not monotonically better.** The 2-bit thermometer beats the 4-bit one by 0.0016 and
beats every other s-held arm. Doubling the input width doubles the literals a clause must manage
under a fixed `L`, and past some point that costs more than the resolution buys.

**4. An accidental hyperparameter finding.** `s` = 25 beats `s` = 6 at width 3,136 by +0.0025 with
everything else identical. `S` = 125 was tuned at width 784; the scaling rule `s = width/S` does not
mean the *same* `s` is right at a different width. Worth knowing before anyone treats `S` as
width-invariant.

## What is NOT tested here

The claim that motivated this track is specifically about **fixed convolutional kernels** applied at
booleanization, which FPTM's Fashion-MNIST result leans on and which the paper explicitly disclaims
as not part of the TM. That is a far stronger intervention than varying threshold count, and it is
untested. Nothing here supports or refutes it. The honest statement is: *threshold count* is worth
little on MNIST, and the convolutional claim remains open.

Also one dataset, one clause count, 20 epochs. A 40-clause model on MNIST at 1 bit already reaches
0.971, so there is limited headroom for an encoder to demonstrate anything.
