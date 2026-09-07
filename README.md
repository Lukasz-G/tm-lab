# tm-lab

A Julia workbench for the Tsetlin Machine stack — TM, Fuzzy-Pattern TM (FPTM), and Graph TM.

Not a port of any one implementation. FPTM and GraphTM each leave semantics undefined at the points
where they would have to meet, so combining them is a new algorithm, and this repo is set up to
treat it as one: decisions before code, gates that can fail, negative results kept.

## Status: scoping complete, no algorithm code yet

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

### Open

Two questions remain, deliberately not guessed at.

1. **Bit- versus symbol-granular fuzziness.** Over thermometer or bag-of-words features a partial
   match is a meaningful sub-pattern. Over a sparse distributed code it may instead be a corrupted
   symbol that aliases to every other symbol sharing a bit. Which one holds decides what the
   evaluator is parameterized over. Answerable first as arithmetic, then as a small toy.
2. **Deliverable form.** One package or several, and whether the model format is pitched for
   adoption by other implementations. Decides package boundaries, so it is a day-one call.

## Layout

```
packages/TMCore/     substrate: evaluator, typed feedback, model format, benchmark harness
research/            experiments and measurement; scripts, not a package
NOTICE.md            third-party attribution, and when a file needs an inline notice
```

One package today. `packages/` exists so that booleanization and graph work can be added as
directories rather than as a refactor — the split into separately registerable packages is open
question 2, and this layout does not pre-empt it.

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
| A | Substrate — evaluator, typed feedback, model format, harness | blocked on open question 1 |
| B | Measurement — satisfied-mask recording, vote histograms, rule mining | can start now, independent of A |
| C | Booleanization — thermometer, convolutional-kernel, n-gram encoders | later |
| D | Algorithmic variants, tested on flat FPTM | after A |
| E | Graph — per-node evaluation, message passing, depth | last |

Track B is deliberately first. FPTM keeps a clause's include set but breaks the identity between
that set and the rule the clause encodes: a clause with 100 literals and `LF`=50 fires on any 50 of
them, which is an m-of-n rule, a much weaker interpretability class. Recording *which* literals
actually match, across many inputs, says whether that structure is recoverable — and it runs against
upstream Tsetlin.jl and a published model, so it returns an answer before the open questions above
are settled. It is falsifiable in both directions, which is the point.

## License

MIT — see [LICENSE](LICENSE). Derives from Tsetlin.jl and reads GraphTsetlinMachine, both MIT;
attribution obligations are in [NOTICE.md](NOTICE.md).
