# How a clause's vote decides which feedback rule it receives.
#
# FPTM computes an integer clause vote and then, at the feedback boundary, throws the magnitude away:
# both Type I and Type II gate on `vote > 0`. A clause matching 1 of LF=50 gets exactly the same
# treatment as one matching 50. That is per spec — arXiv:2508.08350 §2 states the rule explicitly as
# a way to declare a fuzzy clause failed — so it is a deliberate design choice rather than an
# oversight.
#
# It is also a large amount of discarded signal. On Hnilov's published MNIST model, 91.6% of nonzero
# clause votes are strictly interior; on a model trained here, 93.8%. So in roughly nine firings out
# of ten the magnitude carries information that feedback ignores. That measurement is what promotes
# this from a plausible idea to the first variant worth trying.

"""
    FeedbackPolicy

How the clause vote maps onto the choice of feedback rule. See [`ThresholdFeedback`](@ref) and
[`ProportionalFeedback`](@ref).
"""
abstract type FeedbackPolicy end

"""
    ThresholdFeedback()

The published rule: any nonzero vote counts as a firing, and the magnitude is discarded. This is
what both reference implementations do and what the published numbers were produced with, so it is
the default.
"""
struct ThresholdFeedback <: FeedbackPolicy end

"""
    ProportionalFeedback()

Treat the vote as a degree of match rather than a boolean. A clause receives Type Ia with
probability `vote / ceiling`, and Type Ib otherwise; Type II is accepted with the same probability.

This is a strict generalisation — [`ThresholdFeedback`](@ref) is the limit where the gate is a step
function at zero — and it introduces no new hyperparameter, since the ceiling is already known per
clause.

The consequence worth stating in advance: a clause that matches partially now sometimes gets
*eroded* rather than reinforced, where the published rule would always reinforce it. That should
push clauses to either commit to a pattern or decay, which is either a sharpening effect or a
destabilising one. It is not obvious which, and that is the point of measuring it.
"""
struct ProportionalFeedback <: FeedbackPolicy end

"""
    reinforce_branch(policy, vote, ceiling, rng) -> Bool

`true` to apply Type Ia to this clause, `false` to apply Type Ib. A zero vote always means Type Ib,
under every policy — a clause that did not fire at all cannot be reinforced toward the example.
"""
@inline reinforce_branch(::ThresholdFeedback, vote::Integer, ::Integer, _) = vote > 0
@inline function reinforce_branch(::ProportionalFeedback, vote::Integer, ceiling::Integer, rng)
    vote > 0 || return false
    vote >= ceiling && return true                 # a full match always reinforces
    return rand(rng) * ceiling < vote
end

"""
    reject_branch(policy, vote, ceiling, rng) -> Bool

`true` to apply Type II to this clause. There is no alternative rule: a clause that is not selected
here is simply left alone this round.
"""
@inline reject_branch(::ThresholdFeedback, vote::Integer, ::Integer, _) = vote > 0
@inline function reject_branch(::ProportionalFeedback, vote::Integer, ceiling::Integer, rng)
    vote > 0 || return false
    vote >= ceiling && return true
    return rand(rng) * ceiling < vote
end
