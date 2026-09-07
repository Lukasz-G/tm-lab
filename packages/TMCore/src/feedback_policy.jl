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
# of ten the magnitude carries information that feedback ignores.
#
# MEASURED, AND THE PUBLISHED RULE WINS. Using the magnitude costs 0.4 accuracy points on MNIST
# across three seeds with non-overlapping ranges (research/proportional-feedback/). Splitting the
# change: withholding reinforcement costs -0.0037, adding erosion a further -0.0007. It does sharpen
# clauses -- median size 16 to 14, interior fraction 0.92 to 0.89 -- and the sharpening is what
# costs. Unconditional reinforcement is how a clause accumulates the *redundant* literals that give
# it somewhere to degrade to on a noisy input, which is the property FPTM exists to exploit.
#
# So the magnitude is load-bearing for scoring and counterproductive as a learning signal, and a
# large discarded signal is not automatically a wasted one. These policies stay because the result
# is worth being able to reproduce, not because either is recommended.

"""
    FeedbackPolicy

How the clause vote maps onto the choice of feedback rule.

A firing clause can be reinforced toward the example (Type Ia), eroded away from it (Type Ib), or
left alone. The published rule only ever uses the first two and picks between them by `vote > 0`;
policies that use the magnitude need the third option to be expressible, because "reinforce less
often" and "erode more often" are different changes and conflating them makes a measurement
uninterpretable.
"""
abstract type FeedbackPolicy end

"Feedback action for a Type I clause. Three-valued, so that *not* reinforcing and *eroding* stay distinct."
const FEEDBACK_NONE = 0x00
const FEEDBACK_REINFORCE = 0x01
const FEEDBACK_ERODE = 0x02

"""
    ThresholdFeedback()

The published rule: any nonzero vote reinforces, a zero vote erodes, and the magnitude is discarded.
This is what both reference implementations do and what the published numbers were produced with, so
it is the default.
"""
struct ThresholdFeedback <: FeedbackPolicy end

"""
    ProportionalFeedback()

Treat the vote as a degree of match: reinforce with probability `vote / ceiling`, **erode
otherwise**. Type II is accepted at the same probability.

No new hyperparameter — the ceiling is already known per clause — and [`ThresholdFeedback`](@ref) is
the step-function limit. Note that this makes two changes at once: partial matches are reinforced
less often *and* eroded more often. [`ProportionalIdle`](@ref) separates them.
"""
struct ProportionalFeedback <: FeedbackPolicy end

"""
    ProportionalIdle()

Reinforce with probability `vote / ceiling` as [`ProportionalFeedback`](@ref) does, but leave a
clause that fired and lost the draw **alone** rather than eroding it. A clause that did not fire at
all is still eroded, exactly as in the published rule.

This isolates one edit. Against `ProportionalFeedback` it says whether any accuracy difference comes
from withholding reinforcement or from adding erosion — two effects that the obvious formulation
bundles together.
"""
struct ProportionalIdle <: FeedbackPolicy end

"""
    type_i_action(policy, vote, ceiling, rng) -> UInt8

One of `FEEDBACK_REINFORCE`, `FEEDBACK_ERODE` or `FEEDBACK_NONE`.

A zero vote always erodes, under every policy: a clause that did not fire cannot be reinforced
toward the example, and leaving it untouched would remove the only mechanism by which unused clauses
decay and are recycled.
"""
@inline type_i_action(::ThresholdFeedback, vote::Integer, ::Integer, _) =
    vote > 0 ? FEEDBACK_REINFORCE : FEEDBACK_ERODE

@inline function type_i_action(::ProportionalFeedback, vote::Integer, ceiling::Integer, rng)
    vote > 0 || return FEEDBACK_ERODE
    vote >= ceiling && return FEEDBACK_REINFORCE          # a full match always reinforces
    return rand(rng) * ceiling < vote ? FEEDBACK_REINFORCE : FEEDBACK_ERODE
end

@inline function type_i_action(::ProportionalIdle, vote::Integer, ceiling::Integer, rng)
    vote > 0 || return FEEDBACK_ERODE
    vote >= ceiling && return FEEDBACK_REINFORCE
    return rand(rng) * ceiling < vote ? FEEDBACK_REINFORCE : FEEDBACK_NONE
end

"""
    reject_branch(policy, vote, ceiling, rng) -> Bool

`true` to apply Type II. There is no alternative rule here: a clause not selected is left alone.
"""
@inline reject_branch(::ThresholdFeedback, vote::Integer, ::Integer, _) = vote > 0
@inline function reject_branch(::Union{ProportionalFeedback,ProportionalIdle},
                               vote::Integer, ceiling::Integer, rng)
    vote > 0 || return false
    vote >= ceiling && return true
    return rand(rng) * ceiling < vote
end
