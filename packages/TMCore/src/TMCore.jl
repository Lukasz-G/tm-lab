"""
    TMCore

Substrate for the tm-lab stack: typed feedback architecture, bitpacked clause evaluator, model
format, benchmark harness. Variant-agnostic — strict and fuzzy are two methods on one core, and TA
state is a first-class inspectable object. Everything else in tm-lab is a client of this module.

**No algorithm is implemented yet, deliberately.** See the README for the open questions; the
remaining blocker is whether fuzziness operates at bit or symbol granularity, which decides what the
evaluator is parameterized over.
"""
module TMCore

# Design constraints held for the first real commit, recorded here so they are not quietly lost:
#
#   * Feedback rules are swappable components parameterized by type — `feedback!(::TypeIa, ...)`,
#     fuzzy vs strict, classification vs regression. Dispatch, not branches. Free at runtime in
#     Julia, and miserable to retrofit later.
#   * Clause-to-class binding is likewise a type parameter: one-vs-rest now, coalesced later,
#     neither hardcoded.
#   * The clause vote counts down from a ceiling. That ceiling is a policy parameter, because the
#     FPTM paper and its reference implementation disagree about it: the paper caps at the clause's
#     literal count, the implementation uses `LF` flat. Both must be expressible here or the two
#     are not comparable on one evaluator.
#   * The clause evaluator exposes the satisfied mask as a hook. Upstream it is already computed
#     and then discarded; recovering it is one store, not an instrumentation pass. The measurement
#     track depends on this and nothing else.
#
# Hyperparameter validation that is not optional — each is a silent failure, not a loud one:
#
#   * `LF >= 1`. At `LF == 0` no clause can ever vote, so the include set never grows, every
#     decision margin is exactly zero, and a binary model returns the negative class for every
#     input. It reads as "no signal" rather than "broken". `LF == 1` is the strict TM.
#   * `S` scales with input width — upstream derives a literal-decrement count as `length(x) / S`,
#     so a value tuned at one feature count does not transfer to another. Warn on transfer.
#
# Files that transfer Tsetlin.jl's packed layout or inner loop must carry its MIT notice inline.
# See NOTICE.md.

end # module TMCore
