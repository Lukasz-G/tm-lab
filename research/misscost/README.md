# misscost — does charging a failed literal by its automaton's confidence help?

**Status: answered, negative — and the premise itself turns out to be false.**
`julia --project=. run.jl [epochs]`; output in `results.txt`.

## Question

FPTM charges exactly 1 per failed literal. The design notes called this the "most obvious free
lunch": the model already knows how strongly it believes each literal belongs — an automaton just
over the include threshold is a tentative inclusion, one near `state_max` a confident one — and
evaluation discards that. Charge confident misses double and the vote reflects it, at no new
hyperparameter.

## Result

Three arms, MNIST, published hyperparameters, 30 epochs, 5 seeds.

| arm | mean best | vs published | worse on | epoch cost |
|---|---|---|---|---|
| uniform (published) | 0.9724 | — | — | 20 s |
| weighted, nominal midpoint | 0.9724 | +0.0000 | 0/5 | 143 s |
| weighted, calibrated median | 0.9696 | **−0.0028** | **5/5** | 126 s |

## Why it fails, which is the interesting part

**The confidence gradient does not exist.** Included automaton states, measured across every trained
model here:

```
min 128   median 128   max 136-168        (include_limit 128, state_max 255)
```

The median sits **exactly on the include threshold**. More than half of all included automata are at
the floor, with a thin tail reaching about 168 out of a possible 255. The quantity this variant
wanted to exploit is almost entirely absent.

The mechanism is the growth gate. It freezes *all* reinforcement once a clause exceeds `L`, not just
new promotions — and clauses exceed `L` almost immediately, since `L` is a growth gate rather than a
cap. So an automaton that crosses the include threshold stops climbing at once, while Type Ib
erosion keeps pushing it back down. They pile up at the threshold. (Under `HardCap`, which never
freezes reinforcement, the same automata saturate at 255 — so this distribution is a property of the
budget policy, not of the data.)

Both arms then degenerate, in opposite directions:

- **Nominal midpoint (191)** is never reached by anything, so the policy silently becomes the uniform
  one. Identical accuracy to four decimal places, at seven times the cost — which is what a silent
  no-op looks like.
- **Calibrated median** computes a threshold of **128**, the include floor itself, because that is
  where the median is. Every included literal then counts as confident, every miss costs 2, and the
  policy is exactly "halve `LF`" wearing a disguise. That costs 0.28 points on every seed.

## What this settles

The premise was wrong, not just the threshold. "Distance from the include/exclude boundary is
already in the model and is discarded at evaluation time" is true of the *representation* and false
of the *trained values*: reference training does not produce a spread of automaton confidence to
discard.

Any future attempt at this needs to change training first so that confidence accumulates — which
means changing the growth gate, which `budget-paths` shows is load-bearing for accuracy in its own
right. That is a much larger change than the "free lunch" framing suggested, and it is entangled
with a mechanism already known to matter.

There is also a standing cost regardless of accuracy: weighting by automaton state requires
per-automaton lookup instead of a popcount, which is **~7x slower** (20 s to ~140 s per epoch here),
and requires the automata at scoring time, so inference-only models — 8x smaller — cannot use it.

## Kept

`ConfidenceWeightedMissCost` and `calibrate_misscost!` stay in the package. The result is worth being
able to reproduce, and the policy is the natural place for a future attempt to hang off if the
training dynamics are ever changed. `UniformMissCost` remains the default and would remain it even
had the accuracy come out flat, on the inference-size argument alone.
