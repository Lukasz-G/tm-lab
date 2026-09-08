# partial-freeze — the failed attempt that located the mechanism

**Status: failed, and kept because the failure is the finding.** Superseded by
[`eviction/`](../eviction/).

## The idea

In a reference-trained FPTM, included automata sit at exactly the include threshold, so there is no
confidence gradient for any variant to read. `GrowthGate` is the cause: it freezes *all* Type Ia
reinforcement once a clause exceeds `L`, and Type Ia is the only force that can raise an
already-included automaton.

`CapTypeIa` had already unfrozen it and cost 5.7 points, but it also rationed promotions while the
clause was still *under* `L` — a restriction `GrowthGate` does not have. That looked like the
confound responsible.

So `PartialFreeze` unbundled them cleanly: under `L`, behave exactly like `GrowthGate` (unlimited
promotions); over `L`, reinforce only already-included literals and allow no new ones.

## Result

| policy | MNIST accuracy | included-state median | above threshold |
|---|---|---|---|
| GrowthGate (reference) | 0.9712 | 128 | 6.5% |
| **PartialFreeze** | **0.8021** | 255 | 98.7% |
| CapTypeIa (prior test) | 0.9100 | 255 | 99.4% |

It produced the gradient and **cost 16.9 points** — three times worse than the version it was meant
to improve on.

## Why, and what it bought

Type Ib erodes one step per hit, so a literal at 255 needs 127 hits to fall out of the include set
where one at 128 needs a single hit. Clauses become roughly 127x more rigid and stop being
recyclable. **Confidence and forgetting are the same dial** — but only because eviction is a
fixed-size step.

That diagnosis is what made [`eviction/`](../eviction/) possible, where changing eviction to a reset
delivers the same gradient for 0 to 0.4 points instead of 16.9. The hypothesis here was wrong; the
autopsy was the useful part.
