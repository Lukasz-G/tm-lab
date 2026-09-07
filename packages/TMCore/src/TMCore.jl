"""
    TMCore

Substrate for the tm-lab stack: typed feedback architecture, bitpacked clause evaluator, model
format, benchmark harness. Variant-agnostic — strict and fuzzy are two methods on one core, and TA
state is a first-class inspectable object. Everything else in tm-lab is a client of this module.

**No algorithm is implemented yet, deliberately.** The clause-vote ceiling is unresolved: does a
clause's vote max out at `LF`, or at `min(n_literals, LF)`? That is the unit of the vote, so `T`
calibration, vote histograms, vote-proportional feedback and every cross-clause comparison are
measured against it. Writing the evaluator before it is settled means writing it twice. See the
README for the open questions.
"""
module TMCore

# Design constraints held for the first real commit, recorded here so they are not quietly lost:
#
#   * Feedback rules are swappable components parameterized by type — `feedback!(::TypeIa, ...)`,
#     fuzzy vs strict, classification vs regression. Dispatch, not branches. Free at runtime in
#     Julia, and miserable to retrofit later.
#   * Clause-to-class binding is likewise a type parameter: one-vs-rest now, coalesced later,
#     neither hardcoded.
#   * The clause evaluator exposes the satisfied mask as a hook. Upstream it is already computed
#     and then discarded; recovering it is one store, not an instrumentation pass. The measurement
#     track depends on this and nothing else.
#
# Files that transfer Tsetlin.jl's packed layout or inner loop must carry its MIT notice inline.
# See NOTICE.md.

end # module TMCore
