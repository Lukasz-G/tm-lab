# Now that a confidence gradient exists, does anything actually read it usefully?
#
# The chain so far. Confidence-weighted miss cost failed because a reference-trained FPTM has no
# confidence to weight by — every included automaton sits on the include threshold. `eviction/` fixed
# that: partial freeze plus reset eviction produces a real gradient (7% to 94% of automata above the
# threshold on MNIST) for 0 to 0.4 accuracy points.
#
# That is only worth having if something then earns it back. This is the test.
#
#   A  reference                                  GrowthGate, step eviction, uniform cost
#   B  gradient enabled, still uniform cost       isolates what the enabling change costs
#   C  gradient + weighted cost, calibrated       threshold = median of included states
#   D  gradient + weighted cost, nominal 191      the midpoint of the include band, which was
#                                                 unreachable before and is now meaningful
#
# C or D beating B says confidence is worth reading. C or D beating A says the whole chain is a net
# improvement over the published design. Neither beating B says the gradient is decorative, which
# would be the fourth negative result for this family and worth accepting as final.
#
#   julia --project=. research/confidence-payoff/run.jl [mnist|fashion|cifar] [epochs] [seeds]

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

"Type Ib with a selectable eviction rule; `:reset` drops an included literal below the threshold."
function erode!(b, j, s, rng, rule::Symbol)
    il, smin = b.include_limit, b.state_min
    changed = false
    @inbounds for _ in 1:s
        for (arr, mask) in ((b.state, b.included), (b.state_inv, b.included_inv))
            i = rand(rng, 1:b.width)
            v = arr[i, j]
            v > smin || continue
            nv = (rule === :reset && v >= il) ? il - one(eltype(arr)) : v - one(eltype(arr))
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

function update_class_local!(m, ci, x, positive, rng, rule)
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
            erode!(typeI, j, sv, rng, rule)
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

function train_local!(m, X, Y, rng, rule)
    for i in randperm(rng, length(Y))
        yi = findfirst(==(Y[i]), m.classes)
        update_class_local!(m, yi, X[i], true, rng, rule)
        for ci in 1:length(m.classes)
            ci == yi || update_class_local!(m, ci, X[i], false, rng, rule)
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

println("="^90)
@printf("Does the confidence gradient pay for itself?  (%s)\n", uppercase(String(WHICH)))
println("="^90)
@printf("%d clauses/class, T %d, S %d, L %d, LF %d, %d epochs, %d seeds, width %d\n\n",
        CLAUSES, T, S, L, LF, EPOCHS, length(SEEDS), WIDTH)

function arm(budget, rule, mc, calibrate, seed)
    m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF,
                     budget=budget, misscost=mc)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        calibrate && calibrate_misscost!(m)
        train_local!(m, Xtr, Ytr, rng, rule)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
    end
    thr = m.misscost isa ConfidenceWeightedMissCost ?
          Int(TMCore.confidence_threshold(m.misscost, m.positive[1])) : 0
    return (best=best, thr=thr)
end

arms = (
 ("A reference",                     GrowthGate(),    :step,  UniformMissCost(),                     false),
 ("B gradient, uniform cost",        PartialFreeze(), :reset, UniformMissCost(),                     false),
 ("C gradient, weighted calibrated", PartialFreeze(), :reset, ConfidenceWeightedMissCost(),          true),
 ("D gradient, weighted at 191",     PartialFreeze(), :reset, ConfidenceWeightedMissCost(UInt16(191)), false),
)

println("arm                                accuracy           threshold")
println("-"^90)
res = Dict{String,Vector{Float64}}()
for (name, bud, rule, mc, cal) in arms
    accs, info = Float64[], nothing
    for sd in SEEDS
        t = @elapsed (r = arm(bud, rule, mc, cal, sd))
        push!(accs, r.best); info = r
    end
    res[name] = accs
    @printf("%-34s %.4f (sd %.4f)   %4d   (%.0f s/seed)\n",
            name, mean(accs), std(accs), info.thr, 0.0)
end

println()
println("="^90)
A, B = mean(res["A reference"]), mean(res["B gradient, uniform cost"])
for (name, _, _, _, _) in arms
    @printf("%-34s %+.4f vs A     %+.4f vs B\n", name, mean(res[name]) - A, mean(res[name]) - B)
end
println()
println("C or D above B: confidence is worth reading.  C or D above A: the chain beats the")
println("published design.  Neither above B: the gradient is decorative and this family is closed.")
