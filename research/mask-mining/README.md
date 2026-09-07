# mask-mining — is a fuzzy clause recoverable as a rule?

**Status: answered, and it is the first clearly positive result here. Not "k rules plus a tail" as
hypothesised, but a strict conjunctive core plus a tolerance tail — and the core is extractable and
accurate.** `julia --project=. run.jl [n_examples]`; output in `results.txt`.

## Question

FPTM keeps a clause's include set but breaks the identity between that set and the rule the clause
encodes. A clause with 100 literals and `LF`=50 fires on any 50 of them — an m-of-n rule, a much
weaker interpretability class than a conjunction (prior art: Towell & Shavlik, KBANN). The include
set stays printable and a human still learns nothing from reading it.

Since rules are the reason to prefer a TM over a small MLP, this is not a side issue. If FPTM's
50x clause reduction is bought by making clauses unreadable, the field should know.

## Pass/fail, stated in advance

Concentrated masks → the compression is real and recoverable, and the interpretability story is
repairable. Diffuse masks → FPTM trades the family's central selling point for efficiency.
Operationally: do the top 10 distinct satisfied masks cover a majority of a clause's firings?

Run on Hnilov's **published** 40-clause MNIST model, booleanization verified at 0.9755, 10,000 test
examples, all 400 clause slots.

## Result 1 — the literal reading of the criterion: intermediate

Median clause: 23.5 included literals, 859 firings, **195 distinct satisfied masks**, top-1 covering
10%, **top-10 covering 42%**, normalised entropy 0.642.

So a clause is *not* a handful of recurring patterns. Nor is it noise — no clause has top-10 coverage
below 5%, and distinct masks number about one per 4.4 firings rather than one per firing. The stated
criterion returns "intermediate", which on its own would be an unsatisfying answer.

(Ranking by entropy needs care: `H / log(firings)` is biased upward when firings are few — a clause
that fired 8 times cannot exceed 8 distinct masks and scores H=1 by arithmetic. The first pass of
this experiment had exactly that artefact at both ends of its ranking. Rankings here are restricted
to the 241 clauses with at least 500 firings.)

## Result 2 — the structure that is actually there: core plus tail

The median clause has **11 of its 23.5 literals satisfied in at least 95% of its firings**. Roughly
half the include set is a fixed conjunction; the other half varies. That is the real decomposition,
and it is not the one the hypothesis proposed.

So the question becomes: is that core a *rule*? Extract it as a strict conjunction, evaluate it
standalone, and measure lift over the class base rate.

Polarity matters and is easy to misread. A negative clause votes *against* its class, so its core
firing far **below** base rate is the rule working. Lift is reported directionally.

```
  -- positive clauses (core should select the class) --
  class 1 pos clause  1   core 41   fires  740   P(class) 0.995   lift 8.76x
  class 7 pos clause  2   core 38   fires  705   P(class) 0.987   lift 9.60x
  -- negative clauses (core should avoid the class) --
  class 6 neg clause 17   core 29   fires 5888   P(class) 0.004   lift 0.04x
  class 0 neg clause 17   core 12   fires 4964   P(class) 0.003   lift 0.03x
```

Across all 241 clauses with enough firings, not hand-picked ones:

| | n | median lift | quartiles | core size | carries the class signal |
|---|---|---|---|---|---|
| positive | 40 | **7.37x** | 4.70 / 8.79 | 24 of 35 literals | **37 of 40 (92%)** |
| negative | 198 | **0.39x** | 0.13 / 0.90 | 6 of 17 literals | 116 of 198 (59%) |

## What this settles

**The interpretability story is substantially repairable.** A fuzzy FPTM clause decomposes into a
strict conjunction plus a tolerance tail, and for positive clauses that conjunction is a high-precision
rule on its own — 92% of them carry the class signal, at a median 7.4x lift, with the best firing at
99%+ precision on hundreds of examples. That is a readable, checkable rule of the kind the TM family
is chosen for, and it is recoverable from a published model with no retraining.

**But the recovery is asymmetric.** Negative-clause cores are smaller (6 of 17 literals) and carry
the signal only 59% of the time. That is not surprising — "not a 6" is a weaker and more diffuse
concept than "a 1" — but it means half the model is less explicable than the other half, and the
honest summary is "positive clauses are readable, negative ones often are not".

**The original hypothesis was wrong in a productive way.** "This one fuzzy clause is really k rules
plus a tail" predicted a small number of recurring masks. What is there instead is one rule plus
graded tolerance around it. That is a better outcome for interpretability than k rules would have
been: a single conjunction with a robustness margin is easier to read than a disjunction of five.

It also connects the interpretability result to the Track D failures. Those found that removing
tolerance costs accuracy because clauses stop accumulating redundant literals. Here the same
redundancy is visible from the other side: the tail *is* the redundancy, and the core is what it is
redundant around.

## Caveats

One model, one dataset — and the follow-up on IMDb ([`imdb-readability/`](../imdb-readability/))
shows the result does **not** generalise. There the published model's clauses hold ~3,800 literals
with `LF`=64, so they tolerate 1.7% of literals failing against 21% here, the core comes out at 98%
of the clause, and extraction returns a 3,780-term conjunction. The operative quantity is
`LF` / included-literals: this decomposition is informative when that ratio is large and vacuous
when it is small. MNIST cores are also pixel sets, so "readable" here means checkable and
low-entropy rather than semantically meaningful. The 95% core threshold is a
choice; a sweep over it was not done, and the core/tail split would move with it.
