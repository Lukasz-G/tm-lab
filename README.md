# tm-lab

A Julia workbench for the Tsetlin Machine stack — TM, Fuzzy-Pattern TM (FPTM), and Graph TM.

Not a port of any one implementation. FPTM and GraphTM each leave semantics undefined at the points
where they would have to meet, so combining them is a new algorithm, and this repo is set up to
treat it as one: decisions before code, gates that can fail, negative results kept.

## Status: trains, and reproduces published FPTM

Running against Hnilov's **published** 40-clause MNIST model (pipeline validated at 97.55% test
accuracy, so it reproduces the model rather than merely loading it),
[research/vote-histogram/](research/vote-histogram/) answers the first real question: **FPTM's
fuzziness is load-bearing, not decorative.** Of the clause evaluations that vote at all, 91.6% land
strictly inside the interval rather than at the ceiling. So the m-of-n interpretability problem is
real and worth mining — this was falsifiable in the other direction and did not fall that way — and
vote-proportional feedback moves up the queue, since feedback currently binarizes at `vote > 0` and
discards that magnitude in 91.6% of firings.

### Settled: the clause-vote ceiling

A fuzzy clause's vote counts down from a ceiling, one per failed literal, clipped at zero. What that
ceiling is turned out to be the blocking question, because it is the *unit* of the vote — it decides
whether short general clauses vote as loudly as long specific ones, whether `L` and `LF` are
independent knobs, how `T` is calibrated, and whether votes are comparable across clauses at all.

The FPTM paper (arXiv:2508.08350 §2) initializes the vote to the clause's literal count or `LF`,
**whichever is smaller**, with the empty clause special-cased upward to `LF`. Hnilov's own optimized
implementation, Tsetlin.jl, uses **`LF` flat** — no literal-count term at all. The two agree for
empty clauses and for clauses holding at least `LF` literals, and diverge for everything in between,
where the flat form over-votes.

A third source breaks the tie: FuzzyPatternTM, Hnilov's own *reference* implementation, matches the
paper exactly — empty-clause special case and all. Two of three agree; the optimized rewrite is the
outlier, and still is at upstream HEAD.

Measured rather than argued in [research/ceiling-divergence/](research/ceiling-divergence/): on the
published model, 2 of 400 clause slots fall in the divergence band, worth 0.40% of ceiling mass. So
flat `LF` is a safe fast path at convergence, and the paper's ceiling matters during training and
for degenerate clauses. The paper is normative; the ceiling is a policy parameter so both stay
reproducible on one evaluator.

That experiment also turned up something not being looked for: **`L` is a growth gate, not a cap.**
It is checked before an increment pass that pushes many automata over the include threshold at once,
so it decides whether a clause may grow this round rather than bounding its size. On the published
model `L` = 10 while clauses hold 12 to 54 literals — every clause exceeds its own documented cap.

### Settled: bit- versus symbol-granular fuzziness

Over thermometer or bag-of-words features a partial match is a meaningful sub-pattern. Over a sparse
distributed code — a symbol being `H` bits set out of `D` — it might instead be a corrupted symbol
aliasing to every other symbol sharing the remaining bits, which would make bit-level fuzziness
incoherent and force fuzziness to be redefined over whole symbols.

Answered by arithmetic in [research/aliasing/](research/aliasing/). The mechanism is real —
forgiving one bit inflates the false-positive rate by 27x to 763x, worst where the code is sparsest.
But `H` is the lever, not the granularity: the expected number of other symbols that alone drive a
clause to within one of a full vote is 12.2 at `H`=2, `D`=32, and 5.7e-04 at `H`=4, `D`=256. The
danger zone is `H` <= 2 and it ends abruptly.

And the expected fix is not the conservative one. Forgiving a whole symbol turns a 3-symbol clause
into a 2-symbol clause, so at `H` >= 4 symbol-level tolerance admits *more* false positives than
bit-level tolerance of two bits. Symbol granularity buys interpretability, not correctness.

So the evaluator takes a construction-time `(D, H, V)` guard rather than a granularity parameter,
and bit granularity — the cheaper inner loop, with bit-packing intact — is the default. The one case
this does not cover is message symbols, whose codes are cyclic shifts of one another rather than
independent draws.

### Settled: what this repo is

tm-lab is a lab. Code accumulates here, and a standalone registerable package gets extracted if and
when something earns it. The `packages/` layout makes that split a file move rather than a refactor,
so deferring the decision costs almost nothing.

The one exception is the **model format**. Its value is adoption by other implementations, and that
does not get cheaper by waiting — not because the decision is urgent, but because whatever
serialization exists when the question is finally asked is what ends up being formalized. So the
cheap insurance is taken up front: a versioned header plus raw packed arrays, readable from Python
in fifty lines with no Julia runtime. Julia `Serialization` is disqualified — it is fragile across
versions and struct changes, which the measurement track already has to work around.

[TMCore](packages/TMCore/) has the clause evaluator, the three feedback rules and one-vs-rest
training. Bit-packed include masks, a branch-free miss kernel, the satisfied mask exposed as a
first-class output, and — the point of the design — the **clause-vote ceiling and the literal budget
as policy types** rather than constants, because the literature disagrees about both.

Verified three independent ways rather than by smoke test:

- the evaluator matches Hnilov's reference implementation over **4,000,000 clause evaluations with
  zero mismatches** ([tmcore-differential](research/tmcore-differential/));
- the feedback rules match a naive transcription of the reference **state for state**, 13,078 checks
  in the test suite;
- trained from scratch on full MNIST it reaches **0.9769** against the published model's 0.9809, and
  reproduces the characteristic clause-size overshoot that a wrong feedback rule would not
  ([mnist-training](research/mnist-training/)).

That last gap is provenance, not defect — the published model is a merged best-of-512 ensemble
selected on the test set.

## Layout

```
packages/TMCore/     substrate: evaluator, typed feedback, model format, benchmark harness
research/            experiments and measurement; scripts, not a package
NOTICE.md            third-party attribution, and when a file needs an inline notice
```

One package today. `packages/` exists so that booleanization and graph work can be added as
directories rather than as a refactor, and so that pulling any of them out as a standalone package
later stays a file move.

## Getting started

```
julia --project=.        # dev environment; packages/ resolved via [sources], Julia 1.11+
```

```julia
using Pkg; Pkg.test("TMCore")
```

`Manifest.toml` is not committed. `[sources]` in the root `Project.toml` records the package paths,
so a fresh clone resolves without one.

## Tracks

| Track | What | State |
|---|---|---|
| A | Substrate — evaluator, typed feedback, training | reproduces published FPTM; model format and harness still to do |
| B | Measurement — satisfied-mask recording, vote histograms, rule mining | first results in |
| C | Booleanization — thermometer, convolutional-kernel, n-gram encoders | later |
| D | Algorithmic variants, tested on flat FPTM | after A |
| E | Graph — per-node evaluation, message passing, depth | last |

Track B is deliberately first. FPTM keeps a clause's include set but breaks the identity between
that set and the rule the clause encodes: a clause with 100 literals and `LF`=50 fires on any 50 of
them, which is an m-of-n rule, a much weaker interpretability class. Recording *which* literals
actually match, across many inputs, says whether that structure is recoverable — and it runs against
upstream Tsetlin.jl and a published model, so it needs nothing from Track A. It is falsifiable in
both directions, which is the point.

## License

MIT — see [LICENSE](LICENSE). Derives from Tsetlin.jl and reads GraphTsetlinMachine, both MIT;
attribution obligations are in [NOTICE.md](NOTICE.md).
