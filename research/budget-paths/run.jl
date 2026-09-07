# Which growth path does the literal budget L actually need to gate?
#
# Enforcing L as a documented cap costs about 20 accuracy points on a synthetic task. But that
# measurement gated BOTH paths on which a clause can grow — Type Ia reinforcement and Type II
# rejection — while the references gate neither of them by L in Type II. So the number confounds two
# changes and does not say which one hurts.
#
# Four arms, differing only in which paths the budget rations:
#
#   GrowthGate     neither path rationed, and Type Ia additionally freezes once over L  [reference]
#   CapBoth        both paths rationed                                                  [prior result]
#   CapTypeIa      only reinforcement rationed  -- L where the references put the check
#   CapTypeII      only rejection rationed      -- the path the references never check
#
# This matters beyond tidiness. If capping Type II is what hurts, then L was never intended to bound
# clause size and the overshoot is a consequence of design rather than an oversight. If capping
# Type Ia is what hurts, the growth gate is doing real work that a cap would destroy. Either way it
# is the difference between reporting a bug and reporting a design decision.
#
#   julia --project=. research/budget-paths/run.jl [epochs]

include(joinpath(@__DIR__, "..", "mnist.jl"))
using Printf, Random, Statistics
using TMCore
import TMCore: reinforce_allowed, promotion_room

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))
const WIDTH = 784
const CLAUSES, T, S, L, LF = 40, 10, 125, 10, 5

# Experiment-local policies. That these can be defined outside the package, with no change to the
# evaluator or the training loop, is the dispatch design paying for itself.
struct CapTypeIa <: TMCore.LiteralBudgetPolicy end
reinforce_allowed(::CapTypeIa, ::Integer, ::Integer) = true
promotion_room(::CapTypeIa, n::Integer, L::Integer, ::TypeIa) = max(0, Int(L) - Int(n))
promotion_room(::CapTypeIa, ::Integer, ::Integer, ::TypeII) = typemax(Int)

struct CapTypeII <: TMCore.LiteralBudgetPolicy end
reinforce_allowed(::CapTypeII, ::Integer, ::Integer) = true
promotion_room(::CapTypeII, ::Integer, ::Integer, ::TypeIa) = typemax(Int)
promotion_room(::CapTypeII, n::Integer, L::Integer, ::TypeII) = max(0, Int(L) - Int(n))

# The three arms above all drop GrowthGate's reinforcement freeze as well as changing what is
# rationed, so on their own they confound two edits. This one is the clean isolation: the reference
# policy exactly, plus a cap on Type II and nothing else. Comparing it against GrowthGate answers
# the actual question -- would gating Type II by L, as the documentation implies it should be,
# damage the reference model?
struct GateThenCapII <: TMCore.LiteralBudgetPolicy end
reinforce_allowed(::GateThenCapII, n::Integer, L::Integer) = n <= L
promotion_room(::GateThenCapII, ::Integer, ::Integer, ::TypeIa) = typemax(Int)
promotion_room(::GateThenCapII, n::Integer, L::Integer, ::TypeII) = max(0, Int(L) - Int(n))

println("="^78)
println("Which growth path does L need to gate?")
println("="^78)
@printf("MNIST, %d clauses/class, T %d, S %d, L %d, LF %d, %d epochs\n\n",
        CLAUSES, T, S, L, LF, EPOCHS)

tr_px, tr_y, n_tr, _, _ = mnist_train(DATA)
te_px, te_y, n_te, _, _ = mnist_test(DATA)
tr_bits, _ = booleanize_both(tr_px, n_tr)
te_bits, _ = booleanize_both(te_px, n_te)
Xtr = [TMInput(Vector{Bool}(b)) for b in tr_bits]
Xte = [TMInput(Vector{Bool}(b)) for b in te_bits]
Ytr, Yte = Int.(tr_y), Int.(te_y)

function arm(budget, epochs, seed)
    m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF, budget=budget)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:epochs
        train!(m, Xtr, Ytr; rng=rng)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
    end
    c = sort(literal_counts(m))
    return best, c[end÷2], c[end]
end

arms = (("GrowthGate  [reference]", GrowthGate()),
        ("CapBoth", HardCap()),
        ("CapTypeIa   [refs' check point]", CapTypeIa()),
        ("CapTypeII   [refs never check]", CapTypeII()),
        ("GateThenCapII [clean isolation]", GateThenCapII()))

println("policy                             best acc   literals median   max")
println("-"^78)
results = Tuple{String,Float64,Int,Int}[]
for (name, b) in arms
    t = @elapsed ((acc, med, mx) = arm(b, EPOCHS, 20260907))
    @printf("%-34s  %.4f   %10d   %5d   (%.0f s)\n", name, acc, med, mx, t)
    push!(results, (name, acc, med, mx))
end

println()
println("="^78)
base = results[1][2]
for (name, acc, _, _) in results[2:end]
    @printf("%-34s  %+.4f vs reference\n", name, acc - base)
end
