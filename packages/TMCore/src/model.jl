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
    params::Hyperparameters      # not const: set_hyper! rewrites T and LF for schedules
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

"""
    predict(model, X) -> Vector

Predict a batch, across `Threads.nthreads()` threads.

Threading here is free of the tradeoff that kept `train!` serial: scoring reads the clause banks and
writes nothing, so there is no shared mutable state and no race to accept. Each example lands in its
own output slot, so the result is **bit-identical** to the serial version and to itself at any thread
count. Run Julia with `-t auto` to get the threads; at one thread this is the old loop.
"""
function predict(m::TMClassifier{ClassType}, X::AbstractVector{TMInput}) where {ClassType}
    out = Vector{ClassType}(undef, length(X))
    Threads.@threads for i in eachindex(X)
        @inbounds out[i] = predict(m, X[i])
    end
    return out
end

accuracy(predicted::AbstractVector, actual::AbstractVector) = count(predicted .== actual) / length(actual)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

"""
One clause's Type I step. Touches column `j` of `bank` and nothing else, which is what makes the
clause loop safe to run in parallel.

Factored out so the serial and clause-parallel loops execute the *same* code rather than two copies
that drift apart — a divergence here would show up as a silent behaviour difference between
`parallel = :none` and `parallel = :clauses`, which is exactly the bug the split invites.
"""
@inline function _type_i_clause!(m::TMClassifier, bank::ClauseBank, j::Integer, x::TMInput,
                                 update::Real, rng)
    rand(rng) < update || return nothing
    p = m.params
    n = Int(bank.count[j])
    v = clause_vote(bank, j, x, p.LF, m.ceiling, m.misscost)
    # The feedback policy decides how the vote's magnitude is used. Under the published rule it is
    # discarded and any nonzero vote reinforces; under a proportional rule a partial match
    # reinforces only in proportion to how well it matched.
    act = type_i_action(m.feedback, v, ceiling(m.ceiling, n, p.LF), rng)
    if act == FEEDBACK_REINFORCE
        feedback!(TypeIa(), bank, j, x,
                  reinforce_allowed(m.budget, n, p.L),
                  promotion_room(m.budget, n, p.L, TypeIa()))
    elseif act == FEEDBACK_ERODE
        feedback!(TypeIb(), bank, j, p.s, rng)
    end                                            # FEEDBACK_NONE: leave the clause untouched
    return nothing
end

"One clause's Type II step. Same containment as `_type_i_clause!`: column `j` only."
@inline function _type_ii_clause!(m::TMClassifier, bank::ClauseBank, j::Integer, x::TMInput,
                                  update::Real, rng)
    rand(rng) < update || return nothing
    p = m.params
    n = Int(bank.count[j])
    v = clause_vote(bank, j, x, p.LF, m.ceiling, m.misscost)
    if reject_branch(m.feedback, v, ceiling(m.ceiling, n, p.LF), rng)
        # Unrestricted under GrowthGate, which is what both references do; real headroom under
        # HardCap, since Type II grows clauses too and a cap that ignores it is not a cap.
        feedback!(TypeII(), bank, j, x, promotion_room(m.budget, n, p.L, TypeII()))
    end
    return nothing
end

"The two banks of class `ci`, ordered (Type I recipient, Type II recipient)."
@inline function _banks(m::TMClassifier, ci::Integer, positive::Bool)
    return positive ? (m.positive[ci], m.negative[ci]) : (m.negative[ci], m.positive[ci])
end

@inline function _update_fraction(m::TMClassifier, v::Integer, positive::Bool)
    T = m.params.T
    vc = clamp(v, -T, T)
    return (positive ? (T - vc) : (T + vc)) / (2T)
end

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
    update = _update_fraction(m, score(m, ci, x), positive)
    typeI, typeII = _banks(m, ci, positive)
    @inbounds for j in 1:typeI.nclauses
        _type_i_clause!(m, typeI, j, x, update, rng)
    end
    @inbounds for j in 1:typeII.nclauses
        _type_ii_clause!(m, typeII, j, x, update, rng)
    end
    return nothing
end

"""
Minimum work, in **clause-chunk operations per example**, below which `:clauses` runs serially.

A clause count alone cannot express this, which a downstream project found the hard way: their
crossover sat near 1000 clauses per class at width 4,561 where ours sat near 400 at width 10,194.
Same clause count, half the work per clause, same fixed cost per parallel region. The gate has to be
in units of work, so it is `2 · nclasses · nclauses · nchunks` against this constant.

Calibrated against measured break-even at two widths rather than reasoned from first principles, and
it reproduces both: at width 4,561 it keeps 20 and 100 clauses/class serial (measured 0.20× and
0.34×) and enables 500 and 1000 (1.42×, 2.28×); at width 10,194 it keeps 100 serial (1.01×, i.e.
nothing lost) and enables 400 and 1000 (1.92×, 2.86×). Two machines and two widths is thin, so treat
it as a floor that prevents pathological slowdowns rather than a tuned optimum — and measure at a new
width before assuming a mode pays there.
"""
const PARALLEL_MIN_WORK = 32768

"""
Whether `:clauses` should thread at all for this model shape.

Depends only on the model, so it is evaluated once per epoch rather than per example, and never on
`nthreads()` — a gate that moved with the thread count would change results with the launch flag.
"""
@inline function _worth_threading(m::TMClassifier)
    half = m.positive[1].nclauses
    return 2 * nclasses(m) * half * m.positive[1].nchunks >= PARALLEL_MIN_WORK
end

"Smallest number of clauses worth giving a scoring chunk. Without a floor, `_score_chunks` happily
produced one chunk *per clause* on a small bank — 40 tasks to evaluate 40 clauses, which is how the
missing gate below turned into a 5x slowdown rather than a wash."
const MIN_CLAUSES_PER_CHUNK = 16

"""
Chunks per bank for the batched scoring pass. Aimed at a few chunks per thread so the work divides
evenly, floored so no chunk is trivially small, and capped by the clause count.

It depends on `nthreads()`, which is safe *here* and nowhere else in this file: the chunks are summed
with integer addition, which is exact and order-independent, so the total is identical at any thread
count. Anything touching the RNG must not take this liberty.
"""
@inline _score_chunks(ncl::Int, half::Int) =
    clamp(cld(4 * Threads.nthreads(), 2 * ncl), 1, max(1, half ÷ MIN_CLAUSES_PER_CHUNK))

"""
Every class's positive and negative vote for one example, computed in a **single** parallel region.

Scoring one bank at a time cost four synchronisation barriers per example on a binary model, and
measurement put that overhead — not memory bandwidth, which was the obvious suspect and was wrong —
at most of the gap between the ideal speedup and the observed one. Splitting every bank of every
class into chunks and running them as one flat region replaces those barriers with one.
"""
function _scores_threaded!(pv::Vector{Int}, nv::Vector{Int}, partial::Vector{Int},
                           m::TMClassifier, x::TMInput, nchunk::Int)
    ncl = nclasses(m)
    half = m.positive[1].nclauses
    LF = m.params.LF
    csz = cld(half, nchunk)
    total = ncl * 2 * nchunk
    Threads.@threads for u in 1:total
        @inbounds begin
            ci = (u - 1) ÷ (2 * nchunk) + 1
            r = (u - 1) % (2 * nchunk)
            b = r < nchunk ? m.positive[ci] : m.negative[ci]
            c = r < nchunk ? r : r - nchunk
            acc = 0
            for j in (c * csz + 1):min((c + 1) * csz, half)
                acc += clause_vote(b, j, x, LF, m.ceiling, m.misscost)
            end
            partial[u] = acc
        end
    end
    @inbounds for ci in 1:ncl
        p = n = 0
        base = (ci - 1) * 2 * nchunk
        for c in 1:nchunk
            p += partial[base + c]
            n += partial[base + nchunk + c]
        end
        pv[ci] = p
        nv[ci] = n
    end
    return nothing
end

"""
One whole example, with **every clause of every class** spread across threads as a single flat work
list.

This is the axis that helps when `parallel = :classes` cannot: a binary model has two classes to
spread and nothing more, but it still has every clause. The unit here is a `(class, polarity,
clause)` triple, and all of them are disjoint — class `ci` touches only `positive[ci]`/`negative[ci]`,
within a class the Type I and Type II recipients are the two *different* banks, and within a bank
clause `j` touches only column `j` and `count[j]`. So a binary model with 500 clauses per polarity
offers 2000 independent units rather than 2.

Flattening matters as much as the axis does. Threading each clause loop separately would spawn four
task groups per example per class; one flat list spawns once and also recovers the class parallelism
that `:classes` gets, instead of choosing between them.

Scores are computed for every class before any update is applied. That is not a synchronisation
concession — `score(m, ci, x)` reads only class `ci`'s own banks, so no class can observe another's
update anyway — it is what lets the whole update phase be one parallel region.

`upd` and `isp` are caller-owned scratch so this does not allocate per example.

`par` gates **both** phases. An earlier version guarded only the update phase and left scoring
unconditional, so a small model fell back to a serial update while still spawning a parallel region
to score — which made the smallest configurations several times *slower* than `:none` instead of
merely no faster. Scoring and updating are two halves of the same decision and must not be gated
separately.
"""
function update_example_clauses!(m::TMClassifier, x::TMInput, y,
                                 pos::Vector{Vector{R}}, neg::Vector{Vector{R}},
                                 upd::Vector{Float64}, isp::Vector{Bool},
                                 pv::Vector{Int}, nv::Vector{Int}, partial::Vector{Int},
                                 nchunk::Int, par::Bool) where {R}
    ncl = nclasses(m)
    half = m.positive[1].nclauses

    if par
        _scores_threaded!(pv, nv, partial, m, x, nchunk)
        @inbounds for ci in 1:ncl
            isp[ci] = y == m.classes[ci]
            upd[ci] = _update_fraction(m, pv[ci] - nv[ci], isp[ci])
        end
    else
        @inbounds for ci in 1:ncl
            isp[ci] = y == m.classes[ci]
            upd[ci] = _update_fraction(m, score(m, ci, x), isp[ci])
        end
        @inbounds for ci in 1:ncl
            typeI, typeII = _banks(m, ci, isp[ci])
            rI, rII = isp[ci] ? (pos[ci], neg[ci]) : (neg[ci], pos[ci])
            for j in 1:typeI.nclauses
                _type_i_clause!(m, typeI, j, x, upd[ci], rI[j])
            end
            for j in 1:typeII.nclauses
                _type_ii_clause!(m, typeII, j, x, upd[ci], rII[j])
            end
        end
        return nothing
    end

    total = ncl * 2 * half

    Threads.@threads for k in 1:total
        @inbounds begin
            ci = (k - 1) ÷ (2half) + 1
            r = (k - 1) % (2half)
            typeI, typeII = _banks(m, ci, isp[ci])
            rI, rII = isp[ci] ? (pos[ci], neg[ci]) : (neg[ci], pos[ci])
            if r < half
                j = r + 1
                _type_i_clause!(m, typeI, j, x, upd[ci], rI[j])
            else
                j = r - half + 1
                _type_ii_clause!(m, typeII, j, x, upd[ci], rII[j])
            end
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
    train!(model, X, Y; shuffle = true, rng, parallel = :none)

One epoch. `parallel` picks which axis is spread across threads, and the choice is explicit rather
than inferred because every axis carries its own RNG scheme — inferring it from `nthreads()` would
make results depend on how Julia was launched.

- `:none` (default) — the serial schedule that every number in `research/` was produced with: one
  RNG, consumed across all classes in visit order.
- `:classes` — one task per class. Best when there are many classes.
- `:clauses` — one task per clause, within each class in turn. **The axis for binary and few-class
  models**, where `:classes` has nothing to spread but the clause bank is still large.

**Why not across examples**, which is what the reference does: that races on shared automata. It is
defensible for a stochastic learner, but it makes runs irreproducible, and it is not the only
parallelism on offer. Both axes here are race-free *structurally*, not by tolerance:
`update_class!(m, ci, …)` touches only `positive[ci]`/`negative[ci]`, and within a bank clause `j`
touches only column `j` and `count[j]`. Nothing is shared, so nothing can race.

The one thing those units did share is the RNG, so each parallel path gives every unit its own
stream, all seeded from `rng` in index order before any task starts. A threaded run is therefore
**bit-identical to itself at any thread count**, which `test_threading.jl` asserts rather than
assumes.

No parallel mode is bit-identical to `:none`, or to another mode: independent streams cannot
reproduce one stream interleaved. Same algorithm, same distribution, different draws — the
difference is of the same kind as changing the seed. That is why the default stays `:none`, since
flipping it would silently move every number already committed in `research/`.

Julia must be started with threads (`julia -t auto`) for any of this to do anything.

**What to expect, measured — not what the thread count suggests.** 16 threads, binary, 3000 rows,
speedup over `:none`, at two input widths:

| clauses/class | width 4,561 `:classes` | `:clauses` | width 10,194 `:classes` | `:clauses` |
|---|---|---|---|---|
| 20 | 1.1× | 1.0× | — | — |
| 100 | 2.0× | 1.0× | 1.9× | 1.1× |
| 400–500 | 2.3× | 1.4× | 2.1× | 1.6× |
| 1000 | 1.9× | 2.0× | 1.8× | **2.5×** |

`:classes` is capped at roughly the class count, so on a binary model it stops near 2× however many
threads there are. `:clauses` overtakes it only once there is real work per example, and **the
crossover moves with input width**: at 4,561 it is still behind at 500 clauses per class where at
10,194 it had already passed. Half the work per clause, same fixed cost per region. Measure at your
own width rather than reading a mode off this table.

Below `PARALLEL_MIN_WORK` the gate makes `:clauses` fall back to serial, which is what keeps the
small configurations at ~1.0× instead of *slower* than `:none`. An earlier version gated only the
update phase and left scoring unconditional, so the smallest shapes spawned a region per example to
score work that then ran serially — 5× slower than `:none` at 20 clauses per class. The guarantee
now is that a wrong mode costs nothing, not that the right one is obvious.

Neither mode approaches 16×, and the reason is structural rather than fixable by tuning. Examples
must be processed in sequence — each update changes the model the next example is scored against —
so only the work *within* one example can be spread. That is a few hundred microseconds, against a
measured ~25 µs per parallel region. Batch *inference* has no such constraint and reaches ~6-12×,
because it threads across examples.

Going faster than this would mean giving up exact sequential semantics — processing a mini-batch per
parallel region, so updates within a batch do not see each other. That is an algorithmic change with
its own accuracy consequences, not a scheduling detail, which is why it is not hidden behind this
keyword.
"""
function train!(m::TMClassifier{ClassType}, X::AbstractVector{TMInput}, Y::AbstractVector{ClassType};
                shuffle::Bool=true, rng=Random.default_rng(),
                parallel::Symbol=:none) where {ClassType}
    length(X) == length(Y) ||
        throw(DimensionMismatch("got $(length(X)) inputs and $(length(Y)) labels"))
    parallel in (:none, :classes, :clauses) ||
        throw(ArgumentError("parallel must be :none, :classes or :clauses, got :$parallel"))
    order = shuffle ? randperm(rng, length(Y)) : collect(eachindex(Y))

    if parallel === :none
        @inbounds for i in order
            train!(m, X[i], Y[i]; rng=rng)
        end
    elseif parallel === :classes
        # Seeds are drawn in class order, so the streams are fixed before any task starts and no
        # draw depends on when a task was scheduled.
        rngs = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:nclasses(m)]
        Threads.@threads for ci in 1:nclasses(m)
            r = rngs[ci]
            cls = m.classes[ci]
            @inbounds for i in order
                update_class!(m, ci, X[i], Y[i] == cls, r)
            end
        end
    else
        # One stream per (class, polarity, clause). Xoshiro rather than MersenneTwister because this
        # allocates one per clause: 32 bytes against ~2.5 KB, which at thousands of clauses is the
        # difference between incidental and worth noticing.
        pos = [[Random.Xoshiro(rand(rng, UInt64)) for _ in 1:m.positive[ci].nclauses]
               for ci in 1:nclasses(m)]
        neg = [[Random.Xoshiro(rand(rng, UInt64)) for _ in 1:m.negative[ci].nclauses]
               for ci in 1:nclasses(m)]
        # Scratch allocated once per epoch rather than per example: at tens of thousands of examples
        # the allocation would otherwise outweigh what threading saves.
        ncl = nclasses(m)
        nchunk = _score_chunks(ncl, m.positive[1].nclauses)
        upd = Vector{Float64}(undef, ncl)
        isp = Vector{Bool}(undef, ncl)
        pv, nv = Vector{Int}(undef, ncl), Vector{Int}(undef, ncl)
        partial = Vector{Int}(undef, ncl * 2 * nchunk)
        # One decision for the whole epoch: it depends only on the model's shape, so re-deciding it
        # per example would be branch noise in the hot loop.
        par = _worth_threading(m)
        @inbounds for i in order
            update_example_clauses!(m, X[i], Y[i], pos, neg, upd, isp, pv, nv, partial, nchunk, par)
        end
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

"""
    set_hyper!(model; T = nothing, LF = nothing) -> model

Change `T` and/or `LF` on a live model, for hyperparameter schedules such as annealing `LF` from
fuzzy toward strict. Everything else is preserved and the usual validation still applies, so `LF = 0`
is rejected here exactly as it is at construction.

Note that the two are coupled and changing one alone is a real change of regime, not a tweak. A
clause's maximum vote is its ceiling, which `LF` bounds, so lowering `LF` shrinks the whole vote
scale; `T` is the threshold that scale is compared against, and arXiv:2508.08350 §2.1 puts the
optimum near `sqrt(CLAUSES/2 * LF)`. Anneal `LF` with `T` held fixed and the model drifts further
from that relation on every step, which is a confound rather than a schedule.
"""
function set_hyper!(m::TMClassifier; T=nothing, LF=nothing)
    p = m.params
    m.params = Hyperparameters(T = T === nothing ? p.T : T,
                               S = p.S,
                               L = p.L,
                               LF = LF === nothing ? p.LF : LF,
                               width = p.width)
    return m
end
