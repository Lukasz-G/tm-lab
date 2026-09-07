# annealed-lf — does annealing `LF` from fuzzy toward strict help?

**Status: answered, negative. Only the endpoint matters, and ending strict is the bad part.**
`julia --project=. run.jl [epochs]`; output in `results.txt`.

## Question

The idea: fuzziness early gives a gradient-like signal that a strict all-or-nothing clause cannot, so
a model should search faster at high `LF`; sharpening toward `LF` = 1 late should then recover the
precision of a strict TM. Fuzzy for the search, strict for the finish.

## Controls

Two things could make a naive test meaningless, so both are arms rather than assumptions.

**`T` calibration.** A clause's maximum vote is its ceiling, which `LF` bounds, so lowering `LF`
shrinks the whole vote scale while `T` — the threshold that scale is compared against — stays put.
arXiv:2508.08350 §2.1 puts the optimum near `sqrt(CLAUSES/2 × LF)`, so annealing `LF` alone walks
away from the recommended relation on every step. One arm rescales `T` as `sqrt(LF)` to stay on it.

**Direction.** If annealing down and annealing *up* land in the same place, the schedule is doing
nothing and only the endpoint matters. So a reversed arm is included.

MNIST, published hyperparameters, 30 epochs, 5 seeds, linear schedule.

## Result

| arm | best (mean) | vs published | worse on | **final** (mean) | end `LF` |
|---|---|---|---|---|---|
| constant `LF`=5 (published) | 0.9724 | — | — | **0.9715** | 5 |
| anneal 5→1, `T` fixed | 0.9678 | −0.0046 | 5/5 | **0.9366** | 1 |
| anneal 5→1, `T` rescaled | 0.9674 | −0.0050 | 5/5 | **0.9533** | 1 |
| anneal 1→5, `T` fixed (reversed) | 0.9721 | −0.0003 | 3/5 | **0.9711** | 5 |

## Reading it

**Only the endpoint matters.** Both arms that end at `LF` = 1 finish around 0.94–0.95; both that end
at `LF` = 5 finish at 0.971. Where they *started* is worth 0.0004. The schedule is not doing
anything — the terminal `LF` is doing all of it.

**Ending strict is the damage, and it is much larger than the headline suggests.** Compare best
against final: the down-annealed arms peak at 0.967 and finish at 0.937. That gap is the model
actively degrading as `LF` falls — the peak was reached early, while `LF` was still high, and
everything after was downhill. Quoting only the best accuracy would have understated this by a
factor of eight.

**It is not a `T` artifact.** Rescaling `T` on the paper's relation recovers about half the *final*
loss (0.9533 against 0.9366) and none of the best-accuracy loss (0.9674 against 0.9678). So
mis-calibration was a real secondary effect and not the cause. The control earned its place.

**Early fuzziness buys nothing either.** The reversed arm starts strict, where the premise says
learning should be hardest, and still ties the constant arm. Whatever advantage high `LF` confers, it
is available at any point rather than needing to be front-loaded.

Clause sizes corroborate: down-annealed models end at median 11 literals, maximum ~20, against 16
and ~50 for constant `LF`. Strict clauses cannot use tolerance, so the redundant literals never
accumulate.

## What this settles

Consistent with [`proportional-feedback/`](../proportional-feedback/), which found that unconditional
reinforcement matters because it accumulates *redundant* literals — ones not needed for the match a
clause already makes, but which give it somewhere to degrade to. `LF` = 1 removes the tolerance that
makes redundancy useful, so the redundancy stops being built and the model gets brittle. Annealing
toward strict is a slow-motion version of the same mistake.

The "fuzzy to search, strict to finish" intuition imports an optimisation-schedule idea into a model
whose fuzziness is not a search aid to be annealed away — it is the representation. Turning it off at
the end does not sharpen the answer, it discards the mechanism.

## Caveats

One dataset, one starting `LF`, linear schedule, 30 epochs. A gentler schedule that stops at `LF` = 2
or 3 rather than 1 was not tried and would likely land between the endpoints — but since the finding
is that the endpoint is all that matters, that is a prediction the table already makes rather than an
open question. `set_hyper!` stays in the package: schedules are cheap to express and someone will
want to try a different one.
