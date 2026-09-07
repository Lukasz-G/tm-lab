# vote-histogram — is FPTM's fuzziness load-bearing or decorative?

**Status: answered. Fuzziness is load-bearing, decisively.** Run `julia run.jl`; output in
`results.txt`.

## Question

A fuzzy clause votes an integer in `[0, ceiling]`. If that distribution is bimodal — mass at 0 and
at the ceiling, nothing in between — then clauses behave like strict TM clauses with extra
arithmetic, `LF` is decorative, and the interpretability cost FPTM pays buys nothing. If the mass
sits in the interior, fuzziness is doing real work and each clause is genuinely an m-of-n rule.

## Pass/fail, stated before running

Bimodal near 0 and near `LF` → `LF` is decorative. Spread → fuzziness is load-bearing.

## Setup

Hnilov's **published** 40-clause MNIST model (`models/tm_optimized_40_fp.tm` in FuzzyPatternTM), so
this measures FPTM as published rather than as reimplemented. `L`=10, `LF`=5, `T`=10, `S`=125, 10
classes, 400 clause slots. MNIST test set, 10,000 images, booleanized at the upstream threshold
(`x > 0.25`, one bit per pixel).

The booleanization orientation is decided **by accuracy, not by assumption** — a wrong transpose
permutes every bit silently. Raw layout scores 0.9755 on 2,000 test images, transposed scores
0.1395, and the script aborts if neither clears 90%. That accuracy also confirms the whole pipeline
reproduces the published model rather than merely loading it.

## Answer

```
  vote 0 :   3346592   83.66%
  vote 1 :    182400    4.56%
  vote 2 :    163199    4.08%
  vote 3 :    145972    3.65%
  vote 4 :    107091    2.68%
  vote 5 :     54746    1.37%      <- the ceiling
```

**Not bimodal — monotonically decaying.** There is no spike at the ceiling; if anything the ceiling
is the *least* populated nonzero value.

The decisive number: of the 16.34% of evaluations that produce a nonzero vote, only **8.38% sit at
the ceiling** and **91.62% are strictly interior**. When an FPTM clause fires at all, it almost
always fires *partially*.

Per clause, the interior fraction has median **0.936** and mean 0.913. Exactly **2 of 400** clause
slots are effectively strict — and those two are the single-literal clauses found by
[`ceiling-divergence/`](../ceiling-divergence/), whose ceiling is 1, so an interior is impossible
for them by construction. Every clause that *can* vote partially, does.

The 83.66% zero rate is not evidence against this: each clause belongs to one class, so nine tenths
of the test set is out-of-class for it.

## What this settles

**The "LF is decorative" hypothesis is dead.** FPTM's graded clause output is real, so:

- The m-of-n interpretability problem is real too, and worth the mining step. This was falsifiable
  in the other direction and did not fall that way.
- **Vote-proportional feedback moves up the queue.** Feedback currently binarizes at `vote > 0`
  (per spec, deliberately) and so discards a magnitude that carries information in 91.6% of firings.
  That is a much larger discarded signal than expected.
- A strict TM at the same clause count could not reproduce this behaviour, which is the mechanism
  behind FPTM's clause-count reduction.

**The satisfied mask carries structure.** Literal satisfaction frequency within a clause has a
median spread (max − min) of **0.638** — some included literals match nearly always, others rarely.
A diffuse mask would have shown near-zero spread. So the clause has a recoverable core plus a tail,
which is exactly the decomposition the mining step is meant to extract.

## Next in this line

Frequent-itemset mining over the satisfied masks, to turn "core plus tail" into named sub-rules.
That is now justified by the spread measurement rather than assumed. Order it after the marginal
frequencies, per the cheapest-first rule.
