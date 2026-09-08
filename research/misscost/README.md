# misscost

**Q.** Charge a failed literal by its automaton's confidence?

**A.** No, and the premise is false. Included automata sit at **median 128 of a possible 255** —
exactly the include threshold — because the growth gate freezes reinforcement almost immediately.

The nominal threshold (191) is never reached and the policy silently becomes the uniform one. A
calibrated threshold lands on the include floor, making the policy "halve `LF`" in disguise: -0.0028
on 5/5 seeds. Also ~7x slower and unusable on inference-only models.
