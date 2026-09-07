# research/

Experiments, measurement, and one-off analysis. **Not a package** — scripts here may depend on
anything, including upstream Tsetlin.jl, and are not held to the API stability or test coverage
expected of `packages/`.

Each experiment gets a directory with its own `Project.toml` and a `README.md` stating the question,
the pass/fail criterion **decided in advance**, and the answer once it has one. A negative result is
written up here, not deleted.

## Queued

- **`aliasing/`** — closed-form expected spurious-match count for a symbol missing one bit, over
  vocabulary `V`, hypervector bits `H`, dimension `D`. Arithmetic, roughly an hour, no model
  training. Decides whether fuzziness has to operate at symbol granularity rather than bit
  granularity, and therefore what the evaluator is parameterized over. Follow with a 5-10 symbol
  toy only if the arithmetic does not already settle it. Pass/fail stated in advance: bit-granular
  matching degrades toward chance as `V` grows at fixed `H`, while symbol-granular tracks strict.

- **`vote-histogram/`** — run against upstream Tsetlin.jl and Hnilov's published IMDb
  one-clause-per-class model. Checkpoint zero is whether the model loads at all under Julia 1.11.9;
  `save`/`load` are Julia `Serialization` and fragile across versions and struct changes. Then, in
  order of cost: per-clause vote histogram, marginal satisfied-frequency per included literal, and
  only then frequent-itemset mining over the satisfied masks. Bimodal near 0 and near `LF` means the
  fuzziness is decorative; spread means it is load-bearing.

- **`ceiling-divergence/`** — cheap, and shares a loaded model with the above. The FPTM paper caps a
  clause's vote ceiling at its literal count; Tsetlin.jl uses `LF` flat. The two differ only for
  clauses holding between 1 and `LF - 1` literals, so **count how many clauses are in that band**,
  and at what point in training they leave it. The published IMDb configuration has `L` = `LF` = 64
  at one clause per class, which puts a fully grown clause exactly at the boundary — so the gap is
  predicted to be a training-time phenomenon that closes at convergence. That prediction is the
  pass/fail criterion. If it holds, the flat form is a safe fast path and the paper's form matters
  only for transient behaviour; if clauses sit below `LF` at convergence, the two implementations
  are not reproducing the same model and every published number needs re-reading.

## Data

Datasets and downloaded models live in `/data/` and `/models/` at the repo root, both gitignored.
Nothing large or externally sourced gets committed; record provenance in the experiment's README
instead.
