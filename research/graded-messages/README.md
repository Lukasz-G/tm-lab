# graded-messages

**Q.** Binary messages threshold a clause vote at zero and throw the magnitude away. Thermometer-
encoding the vote into `k` levels is the fuzzy/graph combination's most-cited unexplored idea. Does
it pay?

Only asked properly at **equal channel width**. Graded messages multiply message literals by `k`, so
24 binary bits against 24 graded bits from 24 clauses is a 24-bit channel against a 96-bit one, and
the wider channel wins for reasons that have nothing to do with grading. Every arm here sends
exactly 24 bits per edge:

| arm | source clauses | bits each | one bit means |
|---|---|---|---|
| binary | 24 | 1 | clause `j` fired |
| graded `k`=2 | 12 | 2 | clause `j` scored ≥ `t` |
| graded `k`=4 | 6 | 4 | clause `j` scored ≥ `t` |

**A. No, and it gets worse the more you grade it.**

| arm | PREDICATE task | MAGNITUDE task |
|---|---|---|
| no messages **[control]** | 0.5208 | 0.5114 |
| identity messages *[ceiling]* | 1.0000 | 0.9863 |
| **binary** | **1.0000** | **0.9998** |
| graded `k`=2 | 1.0000 | 0.9957 |
| graded `k`=4 | 0.8784 | 0.9766 |

Two tasks, because one is unreadable. A win on a task built to reward magnitude proves nothing, so
there has to be one where graded should lose. **PREDICATE** is Stage 5's adjacency task, where the
useful message is one bit; **MAGNITUDE** has nodes holding 4 symbols and asks whether some node
overlaps pattern P in ≥3 positions while its right neighbour overlaps Q in ≥3, with background nodes
reaching overlap 2 so a single symbol separates the classes. Grading loses on both.

**Why, and it is not the obvious reason.** The first suspect is the ceiling: under `LiteralCapped` a
vote cannot exceed `min(n_included, LF)`, so a bank of two-literal clauses would make levels above 2
unreachable *by construction* and the result would say nothing. Measured, the source clauses hold
**12–16 literals**, so the ceiling is `LF` = 5 and the whole range 1–5 was on offer.

The mean vote of a *firing* source clause is **1.1–1.6**. Votes pile up at the bottom of a range of
five, so thermometer bits 3 and 4 are almost never set — `k`=4 spends 24 bits to send what is
roughly 12 bits' worth, while binary spends 24 bits on four times as many distinct clauses. The
resolution was available and the model did not produce it.

This sits alongside [`vote-histogram`](../vote-histogram/), which found 91.6% of nonzero votes
strictly interior. Both are true: votes are interior *and* concentrated low. Interior votes make
fuzziness load-bearing; low ones make graded messages dead weight.

**Side result worth keeping.** On MAGNITUDE, binary learned messages (0.9998) beat the identity
"ceiling" (0.9863). Identity copies the neighbour's 32 raw bits and leaves the classifier to compute
the overlap; a learned message computes it before sending. Identity is only a ceiling when the useful
message is a copy — which makes this a second, independent task where learning the channel pays, and
here it beats the reference arm rather than a random control.

**Caveats.** Two synthetic tasks, 3 seeds, one round, one `LF`. A vote distribution that concentrates
low is a property of these tasks and this parameter set; grading might pay where votes spread. That
would have to be shown, and the cheap test for it is the vote histogram, not another accuracy table.
