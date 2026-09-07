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
  Cheap addition while the model is loaded: **count clauses carrying fewer than `LF` included
  literals.** The two candidate clause-vote ceilings coincide when every clause has at least `LF`
  literals, so this says empirically whether the open question is reachable in practice.

## Data

Datasets and downloaded models live in `/data/` and `/models/` at the repo root, both gitignored.
Nothing large or externally sourced gets committed; record provenance in the experiment's README
instead.
