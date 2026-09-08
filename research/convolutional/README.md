# convolutional — Stage 3: per-node evaluation, zero message rounds

**Status: gate met. Convolutional FPTM beats flat FPTM by ~2 points at the same clause count.**
First result here that improves on the baseline. `julia --project=. run.jl [epochs] [seeds]
[clauses] [stride]`; output in `results.txt` and `results-stride1.txt`.

## What it is

An image is cut into patches; each patch is a "node"; a clause evaluates on every patch and the
patch outputs combine into one clause output. No messages, so this is Track E's first step with the
graph removed. Position is thermometer-encoded and appended to each patch.

Writing it at all forces two of the four undecided semantics, so they are arms.

## Result — stride 2, 100 patches, 40 clauses/class, 15 epochs, 3 seeds

| arm | accuracy | vs flat |
|---|---|---|
| flat FPTM (baseline) | 0.9505 | — |
| **conv, max, random patch** | **0.9720** | **+0.0216** |
| conv, max, argmax patch | 0.9709 | +0.0205 |
| conv, sum, argmax patch | 0.9683 | +0.0178 |
| conv, max, random, `s` matched | 0.9695 | +0.0190 |
| conv, max, argmax, `s` matched | 0.9711 | +0.0206 |

Stride 1 (361 patches, 8 epochs, 1 seed) goes further: +0.0284 for max/random, +0.0283 for
argmax with `s` matched.

## Stride nearly produced a false negative

The first run used stride 4 — 25 patches — and **convolution lost**:

| stride | patches | best conv vs flat |
|---|---|---|
| 4 | 25 | **−0.0015** |
| 2 | 100 | **+0.0216** |
| 1 | 361 | **+0.0284** |

Matching a pattern "somewhere" is worth nothing when there are only 25 somewheres. A coarse stride
is not a mild efficiency compromise; it removes the mechanism being tested. The conclusion flipped
sign entirely on a parameter chosen for speed.

## The two undecided semantics

**Fuzzy OR across nodes: `max`, not `sum`.** Max beats sum by 0.0038 at stride 2 and 0.0104 at
stride 1. Sum makes the clause output scale with patch count, so it drifts against a fixed `T`.

**Credit assignment: no stable answer.** Random beats argmax at stride 2 (0.9720 vs 0.9709) and
argmax beats random at stride 1 with `s` matched (0.9732 vs 0.9702) and at 160 clauses. The
differences are within a few thousandths and the ordering flips with configuration, so the design
note's expectation that argmax would be the natural fuzzy replacement is neither confirmed nor
refuted. Random — what the reference does — is a safe default.

## Clause efficiency: not supported

Convolution is supposed to be clause-*efficient*, since a patch pattern is reusable across
positions. At stride 4 the opposite held: conv was worst at low clause counts (−0.031 at 10 clauses,
−0.001 at 40, −0.000 at 160). That sweep was run before the stride problem was found and has not
been repeated at stride 2, so it says nothing about the working configuration.

## Caveats

10,000 training images, 15 epochs, patch 10, one dataset. The stride-1 numbers are a single seed.
Absolute accuracy is not a benchmark — published convolutional TMs use far more clauses and data.
The claim is only the comparison against flat FPTM under identical booleanization and clause count.
