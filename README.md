# tm-lab

A Julia workbench for the Tsetlin Machine stack — TM, Fuzzy-Pattern TM (FPTM), and Graph TM.

Not a port of any one implementation. FPTM and GraphTM each leave semantics undefined at the points
where they would have to meet, so combining them is a new algorithm, and this repo is set up to
treat it as one: decisions before code, gates that can fail, negative results kept.

## Status: scoping complete, substrate unblocked, no algorithm code yet

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

The paper is normative. The ceiling becomes a policy parameter so both forms stay reproducible on
one evaluator, and how much the divergence actually costs is left to measurement rather than
argument — see [research/](research/).

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
| A | Substrate — evaluator, typed feedback, model format, harness | unblocked |
| B | Measurement — satisfied-mask recording, vote histograms, rule mining | can start now, independent of A |
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
