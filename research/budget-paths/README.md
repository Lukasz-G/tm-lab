# budget-paths

**Q.** Clauses grow on two paths — Type Ia reinforcement and Type II rejection. Which must `L` gate?

**A.** **Not Type II.** Adding a Type II cap to the reference policy and changing nothing else costs
**13.4 points**; capping Type Ia instead costs 5.7.

Type II is how a clause learns to *reject*. Cap it and clauses never stop matching the wrong class,
keep drawing reinforcement, and grow without bound — max clause size 46 -> 305 -> 774.

So the omission in Type II is working design, not oversight. What is wrong is the name: `L` is
documented as a maximum clause size, is not one, and cannot be made one cheaply.
