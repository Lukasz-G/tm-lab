"""
    TMCore

Substrate for the tm-lab stack: bit-packed clause evaluator, typed feedback architecture, model
format, benchmark harness. Variant-agnostic — strict and fuzzy are one evaluator under different
policies, and clause state is a first-class inspectable object rather than an implementation detail.

**Present state: the evaluator.** Feedback and training are not implemented yet.

Design commitments held here, each cheap now and miserable to retrofit:

  * The clause-vote **ceiling is a policy type**, not a constant. The FPTM paper and Hnilov's
    reference implementation cap it at the clause's own literal count; his optimized rewrite uses a
    flat `LF`. Reproducing either requires being able to express both on one evaluator.
  * The evaluator **exposes the satisfied mask**. It is computed anyway, so recording it costs one
    AND per chunk. Measurement is not an afterthought bolted on later, because recovering which
    literals carried a partial match is the only route back from an m-of-n rule to something a
    human can read. This is load-bearing rather than speculative: on Hnilov's published model,
    91.6% of nonzero clause votes are strictly interior, so partial matching is what these models
    actually do.
  * Hyperparameter validation rejects **silent** failures loudly — `LF = 0`, and hypervector
    encodings too weak for bit-level fuzziness to identify a symbol.
  * Feedback rules will be swappable components parameterized by type, and clause-to-class binding
    likewise, so one-vs-rest and coalesced sharing can coexist without either being hardcoded.

Model format, when it lands: a versioned header plus raw packed arrays, readable from Python in
fifty lines with no Julia runtime. Not Julia `Serialization`, which survives neither a Julia upgrade
nor a struct rename — upstream renamed its central struct in 2026 and broke exactly that way.

Files that transfer Tsetlin.jl's packed layout or inner loop carry its MIT notice inline; see
NOTICE.md.
"""
module TMCore

using Random: Random, randperm

export TMInput,
       ClauseBank, clause_vote, clause_vote!, bank_vote, refresh_counts!, trainable,
       CeilingPolicy, LiteralCapped, FlatLF, ceiling,
       LiteralBudgetPolicy, GrowthGate, HardCap, reinforce_allowed, promotion_room,
       FeedbackRule, TypeIa, TypeIb, TypeII, feedback!,
       FeedbackPolicy, ThresholdFeedback, ProportionalFeedback,
       reinforce_branch, reject_branch,
       ClassBinding, OneVsRest, TMClassifier, vote, score, predict, accuracy,
       train!, update_class!, literal_counts,
       ClauseStats, observe, interior_fraction, satisfaction_spread, vote_histogram,
       BenchResult, benchmark, model_bytes,
       save_model, load_model,
       Hyperparameters, alias_count, check_symbol_encoding, check_transfer

include("input.jl")
include("ceiling.jl")
include("clauses.jl")
include("feedback.jl")
include("feedback_policy.jl")
include("budget.jl")
include("hyperparameters.jl")
include("model.jl")
include("inspect.jl")
include("bench.jl")
include("format.jl")

end # module TMCore
