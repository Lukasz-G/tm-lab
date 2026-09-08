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

- **[`trackd-replication/`](trackd-replication/)** — do the Track D conclusions survive a second
  dataset? **Direction yes, magnitude no.** All five variants lose on Fashion-MNIST too, 3/3 seeds
  each, with the clause-shrinkage mechanism reproducing. But effect sizes move by up to 2x
  (proportional feedback -0.0085 vs -0.0042; Type II cap -0.0615 vs -0.1339), so they are
  dataset-specific and should not be quoted as properties of FPTM. Caveat stated plainly: two 28x28
  grayscale image sets is a weak replication.

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

- **[`booleanization/`](booleanization/)** — how much accuracy lives in the encoder? **On MNIST,
  about +0.002 — and more than half of what a naive test would credit to it is a confound.** `s` is
  derived as `width/S`, so widening 784 → 3136 bits quadruples the forgetting rate unless `S` is
  scaled. Holding `s` fixed, four bits per pixel beats one by +0.0021; letting `s` drift adds
  another +0.0025. Also: fitted thresholds do *not* beat upstream's hardcoded quartiles, and a
  2-bit thermometer beats every 4-bit arm. The convolutional-kernel claim that motivated the track
  is a much stronger intervention and remains untested.

- **[`imdb-readability/`](imdb-readability/)** — do the extracted cores read as rules when literals
  are words? **No, and the reason is a ratio.** Reproduces the published IMDb number (0.9012 vs
  0.9015) at one clause per class, then finds cores of 3,375-4,127 literals: precise (0.979) and
  unreadable. The clause holds 3,843 literals with `LF`=64, so it tolerates 1.7% of them failing and
  is already a near-strict conjunction — against 21% on MNIST, where the decomposition worked. So
  **`LF` / included-literals is the diagnostic**: below a few percent there is no tail to separate.
  Unplanned: **not one core requires a term to be present** — the model is a pure blacklist,
  classifying by which n-grams are missing.

- **[`mask-mining/`](mask-mining/)** — is a fuzzy clause recoverable as a rule? **Yes, and this is
  the first clearly positive result here.** Not "k rules plus a tail" as hypothesised — the median
  clause has 195 distinct satisfied masks and top-10 coverage of only 42% — but a **strict
  conjunctive core plus a tolerance tail**: 11 of 23.5 literals are satisfied in >=95% of firings.
  Extracted standalone, that core is a real rule. Positive-clause cores carry the class signal in
  **92%** of cases at a median **7.37x** lift, the best firing at 99%+ precision over hundreds of
  examples. Negative cores are smaller and hold only 59% of the time, so half the model is markedly
  less explicable than the other half.

- **[`annealed-lf/`](annealed-lf/)** — does annealing `LF` fuzzy-to-strict help? **No, and the
  schedule is irrelevant: only the endpoint matters.** Both arms ending at `LF`=1 finish near 0.94,
  both ending at `LF`=5 finish at 0.971, regardless of where they started. Best-vs-final exposes the
  real damage — down-annealed models peak at 0.967 early and degrade to 0.937 as `LF` falls, so
  quoting best accuracy alone would understate it eightfold. Rescaling `T` on the paper's relation
  recovers half the final loss and none of the peak loss, so mis-calibration was secondary, not
  causal. A reversed (strict→fuzzy) arm ties the baseline, so early fuzziness buys nothing either.

- **[`misscost/`](misscost/)** — does charging a failed literal by its automaton's confidence help?
  **No, and the premise is false.** Included automata sit at median 128 of a possible 255 — exactly
  the include threshold — because the growth gate freezes reinforcement almost immediately, so the
  confidence gradient the idea wanted to exploit barely exists. The nominal threshold is never
  reached and silently becomes a no-op; the calibrated one lands on the include floor, making the
  policy "halve `LF`" in disguise, which costs 0.28 points on 5/5 seeds. Also ~7x slower and
  unusable on inference-only models.

- **[`proportional-feedback/`](proportional-feedback/)** — does using the clause vote's magnitude
  at the feedback boundary help? **No — it hurts on 10 of 10 seeds.** 0.9724 published rule vs
  0.9684 and 0.9682 for the two proportional variants, at roughly 20 standard errors. Isolating the
  two edits: withholding reinforcement costs -0.0040, adding erosion a further -0.0002, so
  essentially all the damage is the withholding. It *sharpened* clauses as hoped (median 16 to 14, interior 0.92 to
  0.89) and that is exactly what cost accuracy — unconditional reinforcement is how a clause
  accumulates the redundant literals that let it degrade gracefully. The premise that motivated the
  experiment is inverted: a large discarded signal is not automatically a wasted one.

- **[`budget-paths/`](budget-paths/)** — which growth path does `L` need to gate? **Not Type II.**
  Adding a Type II cap to the reference policy and changing nothing else costs 13.4 accuracy points,
  against 5.7 for capping Type Ia, because Type II is how a clause learns to *reject* — block it and
  clauses grow without bound (max size 46 → 305 → 774). So `L` is a brake on reinforcement growth,
  the references are right never to check it in Type II, and what is wrong is the name: `L` is
  documented as a maximum clause size, is not one, and cannot be made one cheaply.

- **[`format-crosscheck/`](format-crosscheck/)** — is the model format actually a format? A model
  trained in Julia is read back by `tools/read_tmcore.py`, a pure-standard-library Python reader
  written against the spec rather than against the writer, which reproduces **every per-class score
  exactly** on 50 MNIST cases. Exercises chunk padding at width 784, array ordering, the ceiling
  policy code, and the stored-count consistency check — none of which a self-round-trip would test.

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

- **message-symbol aliasing** — the residual from `aliasing/`. Cyclic-shift binding correlates
  message codes, so the independence assumption fails and the existing result is optimistic there.
  Needed before message passing, not before.

## Data

Datasets and downloaded models live in `/data/` and `/models/` at the repo root, both gitignored.
Nothing large or externally sourced gets committed; record provenance in the experiment's README
instead.
