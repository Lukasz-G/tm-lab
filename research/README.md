# research/

Experiments, measurement, and one-off analysis. **Not a package** — scripts here may depend on
anything, including upstream implementations, and are not held to the API stability or test coverage
expected of `packages/`.

Each experiment gets a directory with a `README.md` stating the question, the pass/fail criterion
**decided in advance**, and the answer once it has one. A negative result is written up here, not
deleted.

Upstream implementations are **cloned on demand, not vendored** — [`upstream.jl`](upstream.jl)
fetches them into `reference/`, which is gitignored. So a fresh checkout reproduces every result
without carrying anyone else's source in our history. [`mnist.jl`](mnist.jl) is the shared dataset
loader, pulling idx files straight from a mirror so no experiment needs a heavy data dependency.

## Answered

- **[`aliasing/`](aliasing/)** — bit- versus symbol-granular fuzziness over a sparse distributed
  code. The criterion passes at `H` <= 2 and fails at `H` >= 4, so `H` is the lever rather than the
  granularity; and symbol granularity turns out to be the coarser of the two, buying
  interpretability rather than correctness. The evaluator takes a construction-time `(D, H, V)`
  guard instead of a granularity parameter. Residual: message symbols, whose codes are cyclic shifts
  rather than independent draws.

- **[`vote-histogram/`](vote-histogram/)** — is FPTM's fuzziness load-bearing or decorative?
  **Load-bearing, decisively.** Of the evaluations that vote at all, 91.62% are strictly interior
  and only 8.38% reach the ceiling; per-clause interior fraction has median 0.936. So the m-of-n
  interpretability problem is real and worth mining, and vote-proportional feedback moves up the
  queue because feedback currently discards that magnitude in 91.6% of firings. Literal
  satisfaction spread (median 0.638) says the mask has core-plus-tail structure to recover.

- **[`mnist-training/`](mnist-training/)** — does TMCore reproduce published FPTM end to end?
  Trained from scratch on full MNIST it reaches **0.9769** against the published model's 0.9809, and
  reproduces the characteristic clause-size overshoot (median 16, max 46, under `L`=10) that a wrong
  feedback rule would not. The 0.4-point gap is provenance, not defect: the shipped model is the
  best of 512 checkpoints over 1000 epochs, then a pairwise merge, both selected on test accuracy —
  so it is an optimistically biased number, and merging is also why its clauses are larger.
  `include_limit` and longer training were both ruled out as explanations.

- **[`tmcore-differential/`](tmcore-differential/)** — does TMCore's bit-packed evaluator match
  FuzzyPatternTM's index-list one? **Exactly**: 4,000,000 comparisons on the published model, zero
  mismatches, and the satisfied mask decodes to the naive recomputation with zero disagreements.
  `FlatLF` diverges on exactly 20,000 comparisons — 2 clause slots x 10,000 inputs, precisely the
  band `ceiling-divergence/` predicted — which confirms the ceiling policy reaches the evaluator
  rather than being silently ignored.

- **[`ceiling-divergence/`](ceiling-divergence/)** — how far does Tsetlin.jl's flat-`LF` ceiling
  depart from the paper's `min(n, LF)`? **Prediction held**: 2 of 400 clause slots in the divergence
  band, 0.40% of aggregate ceiling mass, so flat `LF` is a safe fast path at convergence. Unplanned
  finding: `L` is a growth gate rather than a cap — `L`=10 with clauses holding 12 to 54 literals.

## Queued

- **satisfied-mask mining** — frequent-itemset mining over the recorded masks, turning
  "core plus tail" into named sub-rules. Now justified by the spread measurement rather than
  assumed. Extends `vote-histogram/`.

- **message-symbol aliasing** — the residual from `aliasing/`. Cyclic-shift binding correlates
  message codes, so the independence assumption fails and the existing result is optimistic there.
  Needed before message passing, not before.

## Data

Datasets and downloaded models live in `/data/` and `/models/` at the repo root, both gitignored.
Nothing large or externally sourced gets committed; record provenance in the experiment's README
instead.
