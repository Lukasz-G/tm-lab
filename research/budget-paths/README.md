# budget-paths — which growth path does the literal budget `L` need to gate?

**Status: answered. `L` must not gate Type II, and the references are right never to check it
there.** `julia --project=. run.jl [epochs]`; output in `results.txt`.

## Question

Enforcing `L` as the documented "maximum literals per clause" was measured to cost about 20
accuracy points on a synthetic task. But that measurement rationed **both** paths on which a clause
can grow — Type Ia reinforcement and Type II rejection — while both reference implementations gate
neither of them by `L` in Type II. The number therefore confounded two edits and could not say which
one hurt.

This matters for what conclusion is defensible. If capping Type II is what hurts, `L` was never
meant to bound clause size and the overshoot follows from design. If capping Type Ia is what hurts,
the growth gate is doing work a cap would destroy. It is the difference between reporting a bug and
reporting a design decision, which is not a distinction to guess at before writing to an author.

## Setup

MNIST, published hyperparameters (40 clauses/class, `T`=10, `S`=125, `L`=10, `LF`=5), 30 epochs,
one seed, five policies differing only in what the budget rations.

The experiment's policies are defined **in the experiment file**, outside the package, with no
change to the evaluator or the training loop. That the dispatch design supports this is the point of
having it.

## Result

```
policy                             best acc   literals median   max
GrowthGate  [reference]             0.9720           16      46      (40 s)
CapBoth                             0.5616           10      10     (734 s)
CapTypeIa   [refs' check point]     0.9148           20      71      (51 s)
CapTypeII   [refs never check]      0.4161          593     774      (85 s)
GateThenCapII [clean isolation]     0.8381           14     305     (214 s)
```

**The clean isolation is `GateThenCapII`** — the reference policy exactly, plus a Type II cap and
nothing else. It costs **13.4 points**. Capping Type Ia instead (`CapTypeIa`) costs 5.7. So gating
the rejection path is about 2.4x more damaging than gating the reinforcement path, and the original
20-point synthetic figure was dominated by the Type II component.

The mechanism is visible in the clause sizes rather than inferred. Type II's job is to add literals
until a clause stops matching examples of the wrong class. Cap it, and clauses can never learn to
reject — so they keep firing on wrong-class examples, keep receiving Type Ia, and grow without
bound: maximum clause size rises from 46 to 305, and in the variant where Type Ia is also unrestrained
(`CapTypeII`) to 774 literals out of 1568 possible, at which point the model is degenerate and
accuracy falls below half.

Note also that the three middle arms drop `GrowthGate`'s reinforcement freeze along with changing
what is rationed, so only `GateThenCapII` isolates a single edit. The first run of this experiment
lacked that arm and would have supported a sloppier conclusion.

## What this settles

**`L` is a brake on reinforcement growth, not a capacity bound**, and applying it to rejection is
actively harmful rather than merely unnecessary. So the omission in Type II is a design decision
that works, not an oversight.

That reframes the finding worth reporting upstream. Not "the cap does not cap" — which invites a
patch that would make things worse — but: `L` is named and documented as a maximum clause size, is
not one, and cannot be made one without a double-digit accuracy loss. The paper's `LF <= L` guidance
inherits the same problem, since it reads as a statement about capacity.

## Caveats

One seed and one dataset. The direction is large enough (13.4 points against a roughly 0.4-point
seed-to-seed spread seen elsewhere) that seed noise is not a plausible explanation, but the
magnitude should not be quoted as precise. `CapBoth` is also slow — 734 s against 40 s for the
reference — because clauses pinned at exactly `L` literals fire far more often, which is itself a
sign of how degenerate that configuration is.
