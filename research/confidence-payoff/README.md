# confidence-payoff

**Q.** With a real gradient available, does weighting by confidence help?

**A.** No. Gradient + uniform cost 0.9718; gradient + weighted 0.9691 (calibrated) and 0.9704
(nominal). Both below the uniform arm.

The nominal threshold of 191 is now *reachable* rather than the silent no-op it was in `misscost/`,
and it still loses. So the rule was not failing for want of a gradient; it is simply a bad rule.

Fourth negative in this family, and the one that closes it.
