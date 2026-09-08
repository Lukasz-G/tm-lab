# proportional-feedback

**Q.** Feedback gates on `vote > 0` and discards the magnitude, which carries information in ~92% of
firings. Does using it help?

**A.** No. **Loses on 10 of 10 seeds**, -0.0042 at ~20 standard errors.

Three arms, because the obvious formulation makes two edits at once: withholding reinforcement costs
-0.0040, adding erosion a further -0.0002.

It *sharpened* clauses (median 16 -> 14, interior 0.92 -> 0.89) and that is what cost the accuracy.
Unconditional reinforcement is how a clause accumulates redundant literals to degrade to.
