# tm-lab

A Julia workbench for the Tsetlin Machine stack — TM, Fuzzy-Pattern TM (FPTM), and Graph TM.
Decisions before code, gates that can fail, negative results kept.

## Status, honestly

A working substrate, verified against the reference implementation several ways, plus a handful of
undocumented facts about FPTM.

**Two things work.** A convolutional FPTM — per-patch evaluation, no message passing — beats flat
FPTM by **+0.0216** at the same clause count and identical booleanization. And **learned messages
beat a random channel of the same width**, on a synthetic task built so the channel cannot simply
carry raw features. Every *algorithmic* variant tried has lost, one positive interpretability result
was retracted after its control was run, and regression is untouched.

## Packages

```
packages/TMCore/     evaluator, typed feedback, training, model format, measurement, benchmark
packages/TMBoolean/  booleanization encoders; depends on nothing, not even TMCore
research/            experiments; scripts, not a package
tools/               reference Python reader for the model format
docs/                model format specification
```

`TMCore` is checked against Hnilov's FuzzyPatternTM three ways: the evaluator matches over
**4,000,000 clause evaluations with zero mismatches**, the feedback rules match a naive transcription
of the reference **state for state** (13,078 checks), and training from scratch reaches **0.9769** on
MNIST against the published model's 0.9809 — a gap traced to that model being a merged best-of-512
ensemble selected on the test set.

The [model format](docs/model-format.md) is a fixed binary header and raw packed arrays, deliberately
not Julia `Serialization`. A model trained in Julia is read back by
[tools/read_tmcore.py](tools/read_tmcore.py) — pure standard library, written against the spec —
reproducing every per-class score exactly.

The design's point is that decisions the literature disagrees about are **policy types**, not
constants: clause-vote ceiling, literal budget, feedback rule, miss cost, class binding.

## What was found

- **`L` is a growth gate, not a cap.** It gates whether a clause may grow this round; it does not
  bound its size. Clauses run 5–58× over it. Confirmed in Hnilov's own code, and gating the Type II
  path with it costs 13.4 points, so the omission there is working design rather than oversight. The
  paper's `LF ≤ L` guidance reads as a capacity statement and is not one.
- **Automata pile up exactly on the include threshold**, because `L` freezes the only force that can
  raise an included one. So "automaton confidence" barely exists in a trained model. Confirmed in
  Hnilov's code; removing `L` fixes it and destroys accuracy (0.95 → 0.31).
- **`s = width/S` confounds any encoder comparison at differing widths.** In one measurement it
  accounted for more of the apparent effect than the encoder did.
- **The clause-vote ceiling differs** between the paper, the reference implementation and the
  optimized fork. Small in effect (0.40%), but the two are not the same model.
- **Fuzziness is load-bearing**, not decorative: 91.6% of nonzero clause votes are strictly interior.

## What did not work

Five algorithmic variants — vote-proportional feedback, confidence-weighted miss cost, annealed `LF`,
capping `L`, and confidence-weighted evaluation on a repaired gradient. All lose, on
**15 of 15 dataset-variant combinations** across MNIST, Fashion-MNIST and CIFAR-10, every seed.

One mechanism explains all of them: each reduces a clause's tolerance, so it stops accumulating the
redundant literals that let it degrade on noisy input. Effect sizes scale with task difficulty.

Interpretability is unresolved. See [research/](research/).

## What did work

**Convolutional evaluation** — cutting the image into patches, evaluating each clause on every patch
and taking the **max** — beats flat FPTM by +0.0216 at 40 clauses per class (0.9720 vs 0.9505),
rising to +0.028 with a finer stride.

The result nearly came out backwards: at stride 4, giving only 25 patch positions, convolution
*loses*. Matching a pattern "somewhere" is worth nothing when there are few somewheres, so a stride
chosen for speed removed the mechanism under test rather than mildly weakening it.

**Learned messages** reach 1.0000 on 3/3 seeds where a *random* channel of the same width reaches
0.9523 — on a synthetic sequence task where the message worth sending is one bit and carrying the
neighbour's raw symbol would cost 32. The random control is the whole result: against a no-message
baseline the margin looks like 0.49, against a random channel it is 0.048, and only the second
number says anything about learning.

Both results are single synthetic or single-dataset findings. Neither says message passing helps on
a real problem.

## Getting started

```
julia --project=.
```

```julia
using Pkg; Pkg.test("TMCore"); Pkg.test("TMBoolean")
```

`Manifest.toml` is not committed; `[sources]` in the root `Project.toml` records the package paths,
so a fresh clone resolves without one.

## License

MIT — see [LICENSE](LICENSE). Derives from Tsetlin.jl and reads GraphTsetlinMachine, both MIT;
attribution obligations are in [NOTICE.md](NOTICE.md).
