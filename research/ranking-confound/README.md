# ranking-confound

**Q.** Does the clause contribute anything to the "readable rule", or is this just global feature
relevance?

**A.** Mostly the latter. Ranking over all 25,600 literals with the model **ignored entirely**:

| | top-3 | top-10 | top-20 |
|---|---|---|---|
| POSITIVE, clause | 4731 @ 0.768 | 910 @ 0.889 | 556 @ 0.924 |
| POSITIVE, global | 4731 @ 0.768 | 154 @ **0.935** | 2 @ 1.000 |
| NEGATIVE, clause | 1803 @ 0.832 | 38 @ 0.974 | 4 @ 0.750 |
| NEGATIVE, global | 785 @ **0.969** | 22 @ **1.000** | 0 @ — |

Same vocabulary (5-6 of 10), identical top-3 rule on the positive clause, generally higher precision
without the model. Unsurprising: the score is the chi-square relevance that selected these features.

One residual — clause membership keeps literals **jointly satisfiable** (556 documents at 0.924 where
global collapses to 2). The clause is a selector of compatible features, not a source of insight.

The control any future attempt must clear lives here.
