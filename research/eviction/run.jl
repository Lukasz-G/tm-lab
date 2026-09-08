# Can confidence and plasticity coexist if eviction is redesigned?
#
# partial-freeze established the conflict. Letting a full clause keep strengthening its literals does
# create a confidence gradient — 98.7% of included automata rise above the threshold, median 128 to
# 255 — and costs 16.9 accuracy points. The reason is that Type Ib erodes one step per hit, so a
# literal at 255 needs 127 hits to fall out of the include set where one at 128 needs a single hit.
# With s = 6 over 784 positions that is roughly 130 events against 16,500: clauses become ~127x more
# rigid and can no longer be recycled.
#
# So confidence and forgetting are the same dial only because eviction is a fixed-size step. Two
# alternatives break that coupling:
#
#   reset        an eroded included literal drops straight below the threshold. Eviction costs one
#                hit no matter how confident the literal was, so confidence becomes a read-out with
#                no inertia attached.
#   proportional the decrement scales with how far above the floor the state sits. Eviction time
#                grows logarithmically rather than linearly with confidence — partial protection.
#
# Type Ib is reimplemented here rather than in the package: if one of these wins it earns promotion,
# and if none do the package stays as it is. Everything else routes through TMCore's public feedback
# functions, so only the eviction rule differs between arms.
#
#   julia --project=. research/eviction/run.jl [mnist|fashion|cifar] [epochs] [seeds]

include(joinpath(@__DIR__, "..", "mnist.jl"))
include(joinpath(@__DIR__, "..", "cifar.jl"))
using Printf, Random, Statistics
using TMCore, TMBoolean
import TMCore: reinforce_allowed, promotion_room

const WHICH = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :mnist
const EPOCHS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
const NSEEDS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
const SEEDS = (20260908, 11, 12, 13, 14)[1:NSEEDS]
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))
const CLAUSES, T, S, L, LF = 40, 10, 125, 10, 5
const CIFAR_TRAIN = 20_000

struct PartialFreeze <: TMCore.LiteralBudgetPolicy end
reinforce_allowed(::PartialFreeze, ::Integer, ::Integer) = true
promotion_room(::PartialFreeze, n::Integer, L::Integer, ::TypeIa) = n <= L ? typemax(Int) : 0
promotion_room(::PartialFreeze, ::Integer, ::Integer, ::TypeII) = typemax(Int)

# --------------------------------------------------------------------------
# Eviction rules
# --------------------------------------------------------------------------
"""
    erode!(bank, j, s, rng, rule, rate)

`s` random automata from each polarity, eroded according to `rule`:
`:step` subtracts one (the reference), `:reset` drops an included literal to just below the
threshold, `:prop` subtracts a fraction of its height above the floor.
"""
function erode!(b, j, s, rng, rule::Symbol, rate::Float64)
    il, smin = b.include_limit, b.state_min
    changed = false
    @inbounds for _ in 1:s
        for (arr, mask) in ((b.state, b.included), (b.state_inv, b.included_inv))
            i = rand(rng, 1:b.width)
            v = arr[i, j]
            v > smin || continue
            nv = if rule === :reset && v >= il
                il - one(eltype(arr))                      # one hit evicts, whatever the confidence
            elseif rule === :prop
                d = max(1, round(Int, (Int(v) - Int(smin)) * rate))
                eltype(arr)(max(Int(smin), Int(v) - d))
            else
                v - one(eltype(arr))
            end
            arr[i, j] = nv
            if v >= il && nv < il
                n = (i - 1) >> 6 + 1
                mask[n, j] &= ~(one(UInt64) << ((i - 1) & 63))
                changed = true
            end
        end
    end
    changed && TMCore.recount!(b, j)
    return nothing
end

# Mirrors TMCore.update_class! exactly except for the eviction call, so arms differ in one rule only.
function update_class_local!(m, ci, x, positive, rng, rule, rate)
    p = m.params
    Tt, LFv, Lv, sv = p.T, p.LF, p.L, p.s
    v = clamp(score(m, ci, x), -Tt, Tt)
    upd = (positive ? (Tt - v) : (Tt + v)) / (2Tt)
    typeI  = positive ? m.positive[ci] : m.negative[ci]
    typeII = positive ? m.negative[ci] : m.positive[ci]
    @inbounds for j in 1:typeI.nclauses
        rand(rng) < upd || continue
        n = Int(typeI.count[j])
        vote = clause_vote(typeI, j, x, LFv, m.ceiling, m.misscost)
        act = type_i_action(m.feedback, vote, ceiling(m.ceiling, n, LFv), rng)
        if act == FEEDBACK_REINFORCE
            feedback!(TypeIa(), typeI, j, x,
                      reinforce_allowed(m.budget, n, Lv), promotion_room(m.budget, n, Lv, TypeIa()))
        elseif act == FEEDBACK_ERODE
            erode!(typeI, j, sv, rng, rule, rate)
        end
    end
    @inbounds for j in 1:typeII.nclauses
        rand(rng) < upd || continue
        n = Int(typeII.count[j])
        vote = clause_vote(typeII, j, x, LFv, m.ceiling, m.misscost)
        if reject_branch(m.feedback, vote, ceiling(m.ceiling, n, LFv), rng)
            feedback!(TypeII(), typeII, j, x, promotion_room(m.budget, n, Lv, TypeII()))
        end
    end
end

function train_local!(m, X, Y, rng, rule, rate)
    for i in randperm(rng, length(Y))
        yi = findfirst(==(Y[i]), m.classes)
        update_class_local!(m, yi, X[i], true, rng, rule, rate)
        for ci in 1:length(m.classes)
            ci == yi || update_class_local!(m, ci, X[i], false, rng, rule, rate)
        end
    end
end

function load_data()
    if WHICH === :cifar
        Xr, yr = cifar10(DATA, :train); Xs, ys = cifar10(DATA, :test)
        n = min(CIFAR_TRAIN, size(Xr, 1))
        G, Gt = cifar_gray(Xr[1:n, :]), cifar_gray(Xs)
        th = Thermometer(nthresholds=2); fit!(th, G)
        B, Bt = transform(th, G), transform(th, Gt)
        return ([TMInput(Vector{Bool}(view(B, i, :))) for i in 1:size(B, 1)], Int.(yr[1:n]),
                [TMInput(Vector{Bool}(view(Bt, i, :))) for i in 1:size(Bt, 1)], Int.(ys), size(B, 2))
    end
    trp, tryy, ntr, _, _ = mnist_train(DATA; which=WHICH)
    tep, tey, nte, _, _ = mnist_test(DATA; which=WHICH)
    trb, _ = booleanize_both(trp, ntr); teb, _ = booleanize_both(tep, nte)
    return ([TMInput(Vector{Bool}(b)) for b in trb], Int.(tryy),
            [TMInput(Vector{Bool}(b)) for b in teb], Int.(tey), 784)
end

Xtr, Ytr, Xte, Yte, WIDTH = load_data()

println("="^92)
@printf("Eviction redesign on %s — can confidence and plasticity coexist?\n", uppercase(String(WHICH)))
println("="^92)
@printf("%d clauses/class, T %d, S %d, L %d, LF %d, %d epochs, %d seeds, width %d\n\n",
        CLAUSES, T, S, L, LF, EPOCHS, length(SEEDS), WIDTH)

function state_spread(m)
    v = Int[]
    for banks in (m.positive, m.negative), b in banks
        il = b.include_limit
        for j in 1:b.nclauses, i in 1:b.width
            b.state[i, j] >= il && push!(v, Int(b.state[i, j]))
            b.state_inv[i, j] >= il && push!(v, Int(b.state_inv[i, j]))
        end
    end
    isempty(v) && return (0, 0, 0.0)
    sort!(v)
    return (v[end÷2], v[end], count(>(128), v) / length(v))
end

function arm(budget, rule, rate, seed)
    m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF, budget=budget)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        train_local!(m, Xtr, Ytr, rng, rule, rate)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
    end
    c = sort(literal_counts(m))
    smed, smax, sfrac = state_spread(m)
    return (best=best, lmed=c[end÷2], lmax=c[end], smed=smed, smax=smax, sfrac=sfrac)
end

arms = (("reference (gate + step)",      GrowthGate(),   :step,  0.0),
        ("partial freeze + step",        PartialFreeze(), :step,  0.0),
        ("partial freeze + reset",       PartialFreeze(), :reset, 0.0),
        ("partial freeze + prop 0.25",   PartialFreeze(), :prop,  0.25),
        ("partial freeze + prop 0.50",   PartialFreeze(), :prop,  0.50))

println("arm                            accuracy           lits med/max   state med/max   >thr")
println("-"^92)
results = Dict{String,Vector{Float64}}()
for (name, bud, rule, rate) in arms
    accs, info = Float64[], nothing
    for sd in SEEDS
        t = @elapsed (r = arm(bud, rule, rate, sd))
        push!(accs, r.best); info = r
    end
    results[name] = accs
    @printf("%-30s %.4f (sd %.4f)  %4d /%4d    %3d /%3d   %5.1f%%\n",
            name, mean(accs), std(accs), info.lmed, info.lmax, info.smed, info.smax,
            100 * info.sfrac)
end

println()
println("="^92)
base = mean(results["reference (gate + step)"])
for (name, _, _, _) in arms
    @printf("%-30s %+.4f vs reference\n", name, mean(results[name]) - base)
end
println()
println("A win needs both: accuracy at or near the reference AND '>thr' well above its ~6%.")
