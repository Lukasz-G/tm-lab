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

## Findings about FPTM

| | result |
|---|---|
| [`budget-paths`](budget-paths/) | **`L` is a growth gate, not a cap**, and must not gate Type II (−13.4 pts if it does) |
| [`pileup-verify`](pileup-verify/) | **automata pile up on the include threshold**, caused by `L`; removing `L` is catastrophic |
| [`ceiling-divergence`](ceiling-divergence/) | paper, reference and optimized fork disagree on the clause-vote ceiling; worth 0.40% |
| [`booleanization`](booleanization/) | **`s = width/S` confounds any encoder comparison at differing widths** |
| [`vote-histogram`](vote-histogram/) | fuzziness is load-bearing: 91.6% of nonzero votes strictly interior |
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
