# eviction — can confidence and plasticity coexist?

**Status: yes, via reset eviction, at a cost of 0 to 0.4 accuracy points depending on dataset.**
`julia --project=. run.jl [mnist|fashion|cifar] [epochs] [seeds]`; output in `results-*.txt`.

## The problem

In a reference-trained FPTM, included automata sit at exactly the include threshold — median 128 of a
possible 255, with only 7% above it on MNIST. So "how strongly does the model believe this literal"
has no answer, and every variant that wants to read it is dead before it starts: confidence-weighted
miss cost, confidence-based pruning, confidence-ordered rule extraction.

Three forces explain it. Type Ia is the only one that can raise an already-included automaton, and
`GrowthGate` freezes it entirely once a clause exceeds `L` — which happens in the first epoch and
never reverses. Type Ib erodes regardless of state. Type II can only push a literal *up to* the
threshold, never past it. Constant downward pressure, no upward pressure, so states pile up on the
boundary.

`partial-freeze` tried the obvious fix — keep reinforcing already-included literals when the clause
is full. It produced the gradient (98.7% above threshold) and **cost 16.9 points**.

## Why the obvious fix fails

Type Ib erodes one step per hit. A literal at 128 is evicted by a single hit; one at 255 needs 127.
With `s`=6 over 784 positions that is roughly 130 events against 16,500 — clauses become about 127x
more rigid and stop being recyclable. **Confidence and forgetting are the same dial**, but only
because eviction is a fixed-size step.

## Two ways to break the coupling

- **reset** — an eroded included literal drops straight below the threshold. Eviction costs one hit
  whatever the confidence, so confidence becomes a read-out with no inertia attached.
- **proportional** — the decrement scales with height above the floor. Eviction time grows
  logarithmically rather than linearly: partial protection.

## Result

`>thr` is the share of included automata strictly above the include threshold — the gradient itself.

| arm | MNIST (5 seeds) | Fashion (3) | CIFAR (3) | `>thr` MNIST → |
|---|---|---|---|---|
| reference (gate + step) | 0.9713 | 0.8584 | 0.3313 | 6.7% |
| partial freeze + step | −0.1668 | −0.1123 | −0.1708 | 98.2% |
| **partial freeze + reset** | **+0.0003** | **−0.0012** | **−0.0040** | **94.0%** |
| partial freeze + prop 0.25 | −0.0204 | −0.0198 | −0.0517 | 98.5% |
| partial freeze + prop 0.50 | −0.0489 | −0.0340 | −0.0814 | 96.3% |

**Reset works.** It delivers the gradient — 6.7% to 94.0% on MNIST, 17.4% to 86.9% on Fashion, 21.9%
to 69.3% on CIFAR — at a cost of nothing to 0.4 points. That is roughly 40x cheaper than the naive
version.

**It is not free**, and the write-up should not pretend otherwise. MNIST is a genuine tie (+0.0003
against seed sd 0.0011); Fashion loses 0.0012 and CIFAR 0.0040, both outside their seed noise. The
cost grows with task difficulty, the same ordering seen throughout Track D.

**Proportional eviction is the wrong answer** and confirms the mechanism: partial protection against
eviction buys partial rigidity, and costs 2 to 8 points accordingly.

## What it is worth

Nothing on its own. A confidence gradient is only valuable if something reads it, and the variants
that wanted to are exactly the ones that failed *because* it was absent. The honest statement is that
reset eviction converts "confidence-based ideas are impossible here" into "confidence-based ideas
cost 0 to 0.4 points to enable, before they have earned anything back."

Whether any of them then earn it back is a separate question, and the natural next test is
confidence-weighted miss cost — previously dead on arrival — on a reset-trained model.

## Caveats

Type Ib is reimplemented in this experiment rather than in the package; if reset earns promotion it
should move into TMCore behind an eviction policy type. Everything else routes through TMCore's
public feedback functions, so the arms differ in the eviction rule alone. One clause count, one `L`,
20 epochs.
