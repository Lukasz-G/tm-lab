# ceiling-divergence

**Q.** The FPTM paper and its reference implementation cap a clause's vote ceiling at its literal
count; the optimized fork uses flat `LF`. How much does that matter?

**A.** Little, at convergence. 2 of 400 clause slots fall in the band where the rules disagree;
0.40% of aggregate ceiling mass.

Unplanned, and larger: **`L` is a growth gate, not a cap.** `L`=10 with clauses holding 12-54
literals. The check runs *before* an increment pass that promotes many automata at once.
