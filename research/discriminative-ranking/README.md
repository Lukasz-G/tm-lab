# discriminative-ranking

**Q.** Rank literals by `P(satisfied | in class) - P(satisfied | out of class)`?

**A.** The numbers look good — a 3,443-literal clause reduces to 10 literals matching 910 test
documents at 0.889, against frequency and random baselines at chance.

**The interpretation is retracted by `ranking-confound/`.** Ranking the same way with the model
ignored entirely finds the same vocabulary at higher precision. The readable words come from the
dataset, not the clause.
