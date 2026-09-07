# proportional-feedback — does using the vote's magnitude at the feedback boundary help?

**Status: answered, negative. It hurts, consistently, and the damage comes from withholding
reinforcement rather than from adding erosion.** `julia --project=. run.jl [epochs]`; output in
`results.txt`.

## Question

FPTM computes an integer clause vote and then discards its magnitude: both Type I and Type II gate
on `vote > 0`. That is per spec — arXiv:2508.08350 §2 states it explicitly, and the reference
implementation even carries the comment `# Small change: Added > 0`.

Measurement said the discarded signal was large. On Hnilov's published MNIST model 91.6% of nonzero
clause votes are strictly interior; on a model trained here, 93.8%. So in roughly nine firings out
of ten, "how well did this clause match" is known and thrown away. That made this the most promising
of the queued variants, on the strength of an actual measurement rather than an intuition.

## Pass/fail, stated before running

Partial matches would now sometimes be eroded where the published rule always reinforces them, so
clauses should either **sharpen** (fewer, more decisive literals, accuracy up) or **destabilise**
(clauses churn, accuracy down). A null result would also be informative: it would mean the magnitude
is load-bearing for scoring but not for learning.

## Arms

Two policies would have confounded two edits, exactly as the first `budget-paths` run did. The
obvious formulation both *withholds reinforcement* and *adds erosion*, so a third arm separates them.

| arm | firing clause, full match | firing clause, partial match | non-firing clause |
|---|---|---|---|
| `threshold` (published) | reinforce | reinforce | erode |
| `proportional` | reinforce | reinforce w.p. `vote/ceiling`, else **erode** | erode |
| `proportional-idle` | reinforce | reinforce w.p. `vote/ceiling`, else **nothing** | erode |

MNIST, published hyperparameters, 30 epochs, 10 seeds.

## Result

Ten seeds, paired (every arm sees the same seeds, so the per-seed difference removes seed-to-seed
variance — the honest test for a sub-point effect).

| | mean best | sd | vs published | worse on |
|---|---|---|---|---|
| threshold (published) | 0.9724 | 0.00029 | — | — |
| proportional-idle | 0.9684 | 0.00084 | **−0.0040** (se 0.00033) | **10/10 seeds** |
| proportional | 0.9682 | 0.00047 | **−0.0042** (se 0.00018) | **10/10 seeds** |

Both variants lose on every single seed, at roughly 20 standard errors. This is not noise.

**Withholding reinforcement costs −0.0040. Adding erosion on top costs a further −0.0002.** So
essentially *all* the damage is the withholding, and the erosion half — the part that sounded risky
in advance — is negligible.

Secondary measurements: median clause size falls 16 → 14, maximum ~50 → ~35, and the interior
fraction falls 0.92 → 0.89.

## Why it hurts

The secondary measurements say what happened, and it is not what the pass/fail anticipated. The
prediction offered "sharpen" and "destabilise" as the two outcomes. It **sharpened** — median clause
size fell from 16 to 14, maximum from ~50 to ~33, and the interior fraction fell from 0.92 to 0.89 —
and the sharpening is exactly what cost accuracy.

Unconditional reinforcement of any firing clause is how a clause accumulates *redundant* literals:
ones that are not needed for the match it already makes, but which give it somewhere to degrade to
when an input is noisy. Rationing that reinforcement produces leaner, more brittle clauses. The
model becomes less fuzzy in precisely the sense FPTM exists to exploit.

That also explains why the erosion half barely matters. Erosion removes literals a clause has; the
withholding prevents it acquiring them in the first place, and prevention dominates.

## What this settles

**The published `> 0` rule is right, and now there is a reason rather than an assumption.** The
clause vote's magnitude is load-bearing for *scoring* and counterproductive as a *learning signal*.
Those are separable, and FPTM separates them correctly.

The finding also inverts the premise that motivated the experiment. "91.6% of firings discard
information" is true and is not a defect: the information is deliberately discarded, and using it
makes the model worse. A large discarded signal is not automatically a wasted one.

## Caveats

One dataset, one hyperparameter setting, 30 epochs. Ten seeds, and the direction holds on every
one, so the sign is not in doubt; the effect is nonetheless small in absolute terms (0.4 points). Two variants remain untried and could behave differently:
scaling the *update magnitude* rather than the acceptance probability, and applying proportionality
to Type II only — Type II's job is rejection, where a strong wrong-class match arguably should be
pushed harder than a weak one. Neither is likely to reverse the direction, but neither is tested.
