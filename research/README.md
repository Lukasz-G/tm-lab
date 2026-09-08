# research/

Experiments. **Not a package** — scripts here may depend on anything, including upstream
implementations, and are not held to `packages/`'s API stability or test coverage.

Each directory has a `run.jl`, its raw output in `results.txt`, and a `README.md` giving the question
and the answer. Pass/fail criteria are stated in the script header **before** the run; negative
results are kept, not deleted.

Upstream implementations are cloned on demand by [`upstream.jl`](upstream.jl) into gitignored
`reference/`, so a fresh checkout reproduces everything without vendoring anyone's source.
[`mnist.jl`](mnist.jl) loads MNIST and Fashion-MNIST; [`cifar.jl`](cifar.jl) loads CIFAR-10.

## Verification of the implementation

| | result |
|---|---|
| [`tmcore-differential`](tmcore-differential/) | evaluator matches the reference: 4,000,000 comparisons, 0 mismatches |
| [`mnist-training`](mnist-training/) | trains to 0.9769 vs published 0.9809; the gap is the published model's ensembling |
| [`format-crosscheck`](format-crosscheck/) | independent Python reader reproduces every score exactly |
| [`pileup-verify`](pileup-verify/) | TMCore and FuzzyPatternTM agree to within 1pp on automaton distributions |

## Track E — Stage 3

[`convolutional`](convolutional/) — per-node evaluation, no messages. **Beats flat FPTM by +0.0216
at the same clause count** (0.9720 vs 0.9505, stride 2), rising to +0.028 at stride 1. The gate is
met; this is the first result here that improves on the baseline.

Stride nearly produced a false negative: at stride 4 (25 patches) convolution *loses* by 0.0015. A
coarse stride removes the mechanism rather than mildly weakening it.

Of the two undecided semantics it forces: fuzzy OR across nodes should be **max**, not sum; credit
assignment has **no stable answer** — random and argmax trade places across configurations, both
within a few thousandths.

[`message-round`](message-round/) — Stage 4, one message round. On a task built so flat models
provably cannot win (both controls sit at chance), one round solves it **perfectly at a quarter of
the width** the flat positional encoding needs, and that flat arm only reaches 0.9877.

Qualification that matters: these are **identity** messages — a node receives its neighbours'
symbols — which is functionally a 3-gram window. Learned messages, the thing that distinguishes
GraphTM, are Stage 5 and untested. This shows the machinery works end to end, not that graph
structure earns its keep.

[`learned-messages`](learned-messages/) — do **learned** messages beat a fixed channel? Yes, on a
task where the channel is too narrow to carry raw features. Learned reaches 1.0000 on 3/3 seeds at
72 bits of node width, matching the 96-bit identity ceiling; a random channel of the same width
reaches 0.9523 and varies by seed. The random control is what makes this readable — without it,
"learned messages work" is indistinguishable from "any 40-bit channel works".

[`graded-messages`](graded-messages/) — thermometer-encoding the vote into `k` levels, the
most-cited unexplored idea in the fuzzy/graph combination. **It loses**, at equal channel width,
on both a task where the message is a predicate and one built so the message is a magnitude, and
monotonically in `k`. Source clauses hold 12–16 literals so the full 1–5 range was on offer, but
firing votes average 1.1–1.6, leaving the upper thermometer bits dead. A side result: binary
learned messages beat the identity *ceiling* there, because identity copies the neighbour while a
learned message computes.

## Findings about FPTM

| | result |
|---|---|
| [`budget-paths`](budget-paths/) | **`L` is a growth gate, not a cap**, and must not gate Type II (−13.4 pts if it does) |
| [`pileup-verify`](pileup-verify/) | **automata pile up on the include threshold**, caused by `L`; removing `L` is catastrophic |
| [`ceiling-divergence`](ceiling-divergence/) | paper, reference and optimized fork disagree on the clause-vote ceiling; worth 0.40% |
| [`booleanization`](booleanization/) | **`s = width/S` confounds any encoder comparison at differing widths** |
| [`vote-histogram`](vote-histogram/) | fuzziness is load-bearing: 91.6% of nonzero votes strictly interior |
| [`vote-resolution`](vote-resolution/) | the fuzzy vote buys **~3x** the resolution of its binarization, not `LF`x, and does not scale with `LF` |
| [`eviction`](eviction/) | reset eviction decouples confidence from rigidity, for 0–0.4 pts |
| [`aliasing`](aliasing/) | `hypervector_bits` is the lever for symbol fuzziness, not match granularity |

## Variants tried, all negative

[`proportional-feedback`](proportional-feedback/), [`misscost`](misscost/),
[`annealed-lf`](annealed-lf/), [`confidence-payoff`](confidence-payoff/),
[`partial-freeze`](partial-freeze/), and the `L` arm of [`budget-paths`](budget-paths/).
[`trackd-replication`](trackd-replication/) re-runs them on Fashion-MNIST and CIFAR-10: **15 of 15
dataset-variant combinations lose, every seed.** Effect sizes vary fivefold and track task
difficulty, so only the direction is a property of FPTM.

One mechanism covers all of them: each reduces a clause's tolerance, so it stops accumulating the
redundant literals that let it degrade gracefully. Clause size tracks accuracy throughout.

## Interpretability, unresolved

[`mask-mining`](mask-mining/) found a conjunctive core plus tolerance tail on MNIST;
[`imdb-readability`](imdb-readability/) showed it does not generalise, with `LF`/literals as the
diagnostic. [`literal-ranking`](literal-ranking/) and
[`discriminative-ranking`](discriminative-ranking/) tried ranking a clause's literals;
[`ranking-confound`](ranking-confound/) showed the readable words come from the dataset rather than
the model, and **retracted the result**.

The question — what did this model learn that the data alone does not say — is unanswered. Any future
attempt must clear the control in `ranking-confound/`.

## Data

Datasets and downloaded models live in gitignored `/data/` and `/models/`.
