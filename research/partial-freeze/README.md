# partial-freeze

**Q.** `GrowthGate` freezes the only force that can raise an included automaton. Unfreeze it for
already-included literals?

**A.** It produces the gradient (98.7% above threshold) and **costs 16.9 points**.

Type Ib erodes one step per hit, so a literal at 255 needs 127 hits to leave the include set where
one at 128 needs one — roughly 130 events against 16,500. Clauses become ~127x more rigid and stop
being recyclable. **Confidence and forgetting are the same dial** — but only because eviction is a
fixed-size step.

That diagnosis is what made `eviction/` possible. The hypothesis was wrong; the autopsy was useful.
