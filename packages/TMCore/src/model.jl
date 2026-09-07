# Classifier, and the one-vs-rest training schedule.

"""
    ClassBinding

How clauses are bound to classes. `OneVsRest` gives each class its own positive and negative banks;
`Coalesced` would share one clause pool across classes, as GraphTM does. Kept as a type parameter
so neither is hardcoded — at 1-40 clauses per class the sharing argument mostly dissolves, but
Track E needs the other answer and retrofitting it later would be a rewrite.
"""
abstract type ClassBinding end

struct OneVsRest <: ClassBinding end

"""
    TMClassifier

Fuzzy-Pattern Tsetlin Machine. Each class owns a positive and a negative clause bank; a class's
score is `positive_votes - negative_votes` and prediction is the argmax over classes.

The three policy type parameters are the point of this type: ceiling, literal budget and class
binding are all decisions the literature disagrees about or leaves open, and all three are settled
by dispatch rather than by a branch or a rewrite.
"""
mutable struct TMClassifier{ClassType,S<:Unsigned,B<:ClassBinding,C<:CeilingPolicy,
                            P<:LiteralBudgetPolicy,F<:FeedbackPolicy,M<:MissCostPolicy}
    const params::Hyperparameters
    const classes::Vector{ClassType}
    const positive::Vector{ClauseBank{S}}
    const negative::Vector{ClauseBank{S}}
    const ceiling::C
    const budget::P
    const binding::B
    const feedback::F
    misscost::M            # not const: calibrate_misscost! replaces it
end

"""
    TMClassifier(classes, width; clauses_per_class, T, S, L, LF, kwargs...)

`clauses_per_class` is the total across both polarities, matching the reference's `clauses_num`, so
each polarity gets half of it.
"""
function TMClassifier(classes::AbstractVector{ClassType}, width::Integer;
                      clauses_per_class::Integer, T::Integer, S::Integer, L::Integer, LF::Integer,
                      states::Integer=256, include_limit::Integer=128,
                      state_type::Type{ST}=UInt8,
                      ceiling::CeilingPolicy=LiteralCapped(),
                      budget::LiteralBudgetPolicy=GrowthGate(),
                      binding::ClassBinding=OneVsRest(),
                      feedback::FeedbackPolicy=ThresholdFeedback(),
                      misscost::MissCostPolicy=UniformMissCost()) where {ClassType,ST<:Unsigned}
    params = Hyperparameters(T=T, S=S, L=L, LF=LF, width=width)
    cls = collect(sort(unique(classes)))
    length(cls) >= 2 || throw(ArgumentError("need at least two classes, got $(length(cls))"))
    half = clauses_per_class ÷ 2
    half >= 1 || throw(ArgumentError("clauses_per_class must be at least 2, got $clauses_per_class"))
    mk() = [ClauseBank{ST}(width, half; states=states, include_limit=include_limit) for _ in cls]
    return TMClassifier{ClassType,ST,typeof(binding),typeof(ceiling),typeof(budget),
                        typeof(feedback),typeof(misscost)}(
        params, cls, mk(), mk(), ceiling, budget, binding, feedback, misscost)
end

nclasses(m::TMClassifier) = length(m.classes)

"""
    vote(model, ci, x) -> (positive, negative)

Summed clause votes for class index `ci`.
"""
@inline function vote(m::TMClassifier, ci::Integer, x::TMInput)
    LF = m.params.LF
    return (bank_vote(m.positive[ci], x, LF, m.ceiling, m.misscost),
            bank_vote(m.negative[ci], x, LF, m.ceiling, m.misscost))
end

"""
    score(model, ci, x) -> Int

Class score, `positive - negative`. Uncalibrated; its magnitude is a confidence only loosely.
"""
@inline function score(m::TMClassifier, ci::Integer, x::TMInput)
    p, n = vote(m, ci, x)
    return p - n
end

"""
    predict(model, x)

Highest-scoring class. Ties go to the lower class index, matching the reference's strict `>`.
"""
function predict(m::TMClassifier{ClassType}, x::TMInput) where {ClassType}
    best_i, best_v = 1, typemin(Int)
    @inbounds for ci in 1:nclasses(m)
        v = score(m, ci, x)
        if v > best_v
            best_v, best_i = v, ci
        end
    end
    return m.classes[best_i]
end

predict(m::TMClassifier, X::AbstractVector{TMInput}) = [predict(m, x) for x in X]

accuracy(predicted::AbstractVector, actual::AbstractVector) = count(predicted .== actual) / length(actual)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

"""
    update_class!(model, ci, x, positive, rng)

One class's share of one example.

`positive` marks `ci` as the true class. The bank receiving Type I feedback and the bank receiving
Type II swap between the two cases: for the true class, positive clauses are reinforced toward the
example and negative clauses pushed away from it; for every other class, the reverse. That swap is
the whole of one-vs-rest.

Both loops draw their own acceptance probability per clause, as the reference does.
"""
function update_class!(m::TMClassifier, ci::Integer, x::TMInput, positive::Bool, rng)
    p = m.params
    T, LF, L, s = p.T, p.LF, p.L, p.s
    v = clamp(score(m, ci, x), -T, T)
    update = (positive ? (T - v) : (T + v)) / (2T)

    typeI  = positive ? m.positive[ci] : m.negative[ci]
    typeII = positive ? m.negative[ci] : m.positive[ci]

    @inbounds for j in 1:typeI.nclauses
        rand(rng) < update || continue
        n = Int(typeI.count[j])
        v = clause_vote(typeI, j, x, LF, m.ceiling, m.misscost)
        # The feedback policy decides how the vote's magnitude is used. Under the published rule it
        # is discarded and any nonzero vote reinforces; under a proportional rule a partial match
        # reinforces only in proportion to how well it matched.
        act = type_i_action(m.feedback, v, ceiling(m.ceiling, n, LF), rng)
        if act == FEEDBACK_REINFORCE
            feedback!(TypeIa(), typeI, j, x,
                      reinforce_allowed(m.budget, n, L),
                      promotion_room(m.budget, n, L, TypeIa()))
        elseif act == FEEDBACK_ERODE
            feedback!(TypeIb(), typeI, j, s, rng)
        end                                        # FEEDBACK_NONE: leave the clause untouched
    end
    @inbounds for j in 1:typeII.nclauses
        rand(rng) < update || continue
        n = Int(typeII.count[j])
        v = clause_vote(typeII, j, x, LF, m.ceiling, m.misscost)
        if reject_branch(m.feedback, v, ceiling(m.ceiling, n, LF), rng)
            # Unrestricted under GrowthGate, which is what both references do; real headroom under
            # HardCap, since Type II grows clauses too and a cap that ignores it is not a cap.
            feedback!(TypeII(), typeII, j, x, promotion_room(m.budget, n, L, TypeII()))
        end
    end
    return nothing
end

"""
    train!(model, x, y; rng)

One example. The true class is updated, then every other class in turn — the reference visits all
of them rather than sampling one negative class, which costs `nclasses` times more feedback per
example but is what the published numbers were produced with.
"""
function train!(m::TMClassifier{ClassType}, x::TMInput, y::ClassType;
                rng=Random.default_rng()) where {ClassType}
    yi = findfirst(==(y), m.classes)
    yi === nothing && throw(ArgumentError("unknown class $y"))
    update_class!(m, yi, x, true, rng)
    @inbounds for ci in 1:nclasses(m)
        ci == yi || update_class!(m, ci, x, false, rng)
    end
    return m
end

"""
    train!(model, X, Y; shuffle = true, rng)

One epoch. Single-threaded: the reference parallelises across examples and accepts the resulting
races on shared automata, which is defensible for a stochastic learner but makes runs
irreproducible. Determinism is worth more than speed while correctness is still being established.
"""
function train!(m::TMClassifier{ClassType}, X::AbstractVector{TMInput}, Y::AbstractVector{ClassType};
                shuffle::Bool=true, rng=Random.default_rng()) where {ClassType}
    order = shuffle ? randperm(rng, length(Y)) : eachindex(Y)
    @inbounds for i in order
        train!(m, X[i], Y[i]; rng=rng)
    end
    return m
end

"""
    literal_counts(model) -> Vector{Int}

Included-literal count of every clause slot in the model. The distribution against `L` is what shows
whether the literal budget is behaving as a cap or as a growth gate.
"""
function literal_counts(m::TMClassifier)
    out = Int[]
    for banks in (m.positive, m.negative), b in banks
        append!(out, Int.(b.count))
    end
    return out
end

"""
    calibrate_misscost!(model) -> model

Set the model's confidence threshold to the **median state of its included automata**, so that the
policy splits the distribution it actually has rather than the one the state range nominally allows.

Without this the policy is usually a no-op: reference training leaves included automata bunched just
above `include_limit`, far below the nominal midpoint. With it, roughly half of included literals
count as confident by construction, which is what makes the comparison against uniform cost a test
of the idea rather than of a threshold guess.

Call it after training, and again whenever training continues — the distribution moves.
"""
function calibrate_misscost!(m::TMClassifier)
    states = UInt16[]
    for banks in (m.positive, m.negative), b in banks
        b.state === nothing && continue
        st, sti, il = b.state, b.state_inv, b.include_limit
        for j in 1:b.nclauses, i in 1:b.width
            st[i, j] >= il && push!(states, UInt16(st[i, j]))
            sti[i, j] >= il && push!(states, UInt16(sti[i, j]))
        end
    end
    isempty(states) && return m
    sort!(states)
    m.misscost = ConfidenceWeightedMissCost(states[max(1, length(states) ÷ 2)])
    return m
end
