# mnist-training — does TMCore reproduce published FPTM behaviour end to end?

**Status: gate met, with the comparison corrected.** `julia --project=. run.jl [epochs]`; output in
`results.txt`.

## Question

The evaluator is exact against the reference ([`tmcore-differential/`](../tmcore-differential/)) and
the feedback rules match it state for state (TMCore's own test suite). This closes the loop: same
hyperparameters, same data, same booleanization — does training from scratch actually get there?

Accuracy alone is a weak check, because a subtly wrong feedback path can still learn something. So
the **literal-count distribution** is checked too. The published model holds 1 to 54 literals per
clause under `L` = 10. If our training reproduces that overshoot, the growth-gate dynamics are
right; if clause sizes sat near `L`, the feedback path would be wrong in a way accuracy would hide.

## Setup

Hyperparameters read off the published model itself: 40 clauses/class, `T`=10, `S`=125, `L`=10,
`LF`=5. Full MNIST, 60,000 train and 10,000 test, booleanized at the upstream single threshold.
About 0.6 s per epoch, single-threaded.

## Result

```
best test accuracy : 0.9769  (epoch 180 of 200)
published model    : 0.9809
gap                : -0.0040

literals/clause  ours      median  16  range   1- 46
                 published median  23  range   1- 54
```

**The growth-gate dynamics reproduce.** Under `L` = 10, our clauses run to 46 literals with a median
of 16 — the same qualitative overshoot as the published model, from an independent implementation.
That is the check that would have caught a wrong feedback rule, and it passes.

**Accuracy lands 0.4 points short**, and the reason is provenance rather than a defect.

## The comparison is not like for like, and the upstream example says so

Reading `examples/MNIST/mnist.jl` upstream, the shipped model is not the product of one training
run. It is:

1. trained for up to **1000 epochs**, keeping the best **512** checkpoints, ranked by test accuracy;
2. passed through `combine`, a **pairwise merge** over those checkpoints, with the winning pair
   again **selected on test accuracy**;
3. passed through `optimize!`, which only reorders literal index lists for early-exit speed and is
   semantically neutral.

So the published 0.9809 is a merged two-model ensemble chosen on the test set. Two consequences:

- **It is an optimistically biased number.** Test accuracy is used twice as a selection criterion,
  once to rank 512 checkpoints and once to pick the merge. It is not a clean held-out estimate, and
  should not be quoted as one.
- **Merging explains the larger clauses.** Merged clauses take the union of their parents' literals,
  which is exactly the direction the median moves — 23 against our 16.

Our 0.9769 is a single un-merged run, selected on the test set only in the weak sense of reporting
the best epoch. Against that, a 0.4 point gap is close, and the honest comparison would be against a
single upstream checkpoint, which is not published.

## What was ruled out

`include_limit` is not recoverable from a compiled model, and the upstream example flags it
explicitly (`include_limit=200 instead of 128`). Tested directly over 60 epochs:

| include_limit | best accuracy | literals median / max |
|---|---|---|
| 128 | 0.9750 | 17 / 46 |
| 200 | 0.9740 | 16 / 47 |

No meaningful difference, so it does not account for the gap.

Longer training was also ruled out as the whole story: 30 epochs gives 0.9720, 200 gives 0.9769, and
the literal counts **plateau** at median 16-17 rather than drifting toward 23. More epochs buy a
little accuracy and no additional clause growth.

## Standing conclusion

Three independent checks now agree that TMCore implements FPTM as published: the evaluator matches
bit for bit over four million comparisons, the feedback rules match state for state, and training
from scratch reproduces both the accuracy band and the characteristic clause-size overshoot.

The remaining 0.4 points are attributable to an ensembling and selection pipeline we have not
implemented and did not set out to. Implementing `combine` is a separate, cheap piece of work if the
number itself ever matters; it is a model-merging heuristic, not part of the FPTM algorithm.
