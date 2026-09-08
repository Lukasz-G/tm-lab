# Can a clause have both a size limit and a confidence gradient?
#
# In a reference-trained FPTM, included automata sit at exactly the include threshold: median 128 of
# a possible 255. So "how strongly does the model believe this literal" has no answer, and every
# variant that wants to read it — confidence-weighted miss cost, confidence-based pruning,
# confidence-ordered rule extraction — is dead before it starts.
#
# The cause is that Type Ia reinforcement does two jobs at once: it pushes excluded literals TOWARD
# inclusion (growth), and pushes included literals DEEPER (confidence). GrowthGate freezes the whole
# block once a clause exceeds L, so it stops both. Type Ib erodes regardless and Type II can only
# push a literal up TO the threshold, never past it — so nothing else can create a gradient.
#
# CapTypeIa already unfroze confidence and cost 5.7 points. But it changed a second thing: it also
# rationed promotions while the clause was still UNDER L, a restriction GrowthGate does not have.
# That confound is probably what cost the points.
#
# PartialFreeze separates them properly:
#   under L : reinforce everything, unlimited promotions  -- identical to GrowthGate
#   over  L : reinforce only already-included literals, zero new promotions
#
# The freeze becomes partial rather than total. Growth dynamics are untouched exactly where
# GrowthGate acts; the only change is that a full clause keeps strengthening what it already has.
#
#   julia --project=. research/partial-freeze/run.jl [mnist|fashion|cifar] [epochs] [seeds]

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

"The proposed policy: freeze growth at L, never freeze strengthening."
struct PartialFreeze <: TMCore.LiteralBudgetPolicy end
reinforce_allowed(::PartialFreeze, ::Integer, ::Integer) = true
promotion_room(::PartialFreeze, n::Integer, L::Integer, ::TypeIa) = n <= L ? typemax(Int) : 0
promotion_room(::PartialFreeze, ::Integer, ::Integer, ::TypeII) = typemax(Int)

"The already-measured comparison: rations promotions everywhere, including under L."
struct CapTypeIa <: TMCore.LiteralBudgetPolicy end
reinforce_allowed(::CapTypeIa, ::Integer, ::Integer) = true
promotion_room(::CapTypeIa, n::Integer, L::Integer, ::TypeIa) = max(0, Int(L) - Int(n))
promotion_room(::CapTypeIa, ::Integer, ::Integer, ::TypeII) = typemax(Int)

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
    trp, try_, ntr, _, _ = mnist_train(DATA; which=WHICH)
    tep, tey, nte, _, _ = mnist_test(DATA; which=WHICH)
    trb, _ = booleanize_both(trp, ntr); teb, _ = booleanize_both(tep, nte)
    return ([TMInput(Vector{Bool}(b)) for b in trb], Int.(try_),
            [TMInput(Vector{Bool}(b)) for b in teb], Int.(tey), 784)
end

Xtr, Ytr, Xte, Yte, WIDTH = load_data()

println("="^88)
@printf("Partial freeze on %s — can a clause have a size limit AND a confidence gradient?\n",
        uppercase(String(WHICH)))
println("="^88)
@printf("%d clauses/class, T %d, S %d, L %d, LF %d, %d epochs, %d seeds, width %d\n\n",
        CLAUSES, T, S, L, LF, EPOCHS, length(SEEDS), WIDTH)

"Distribution of included automaton states — the quantity the whole question is about."
function state_spread(m)
    v = Int[]
    for banks in (m.positive, m.negative), b in banks
        il = b.include_limit
        for j in 1:b.nclauses, i in 1:b.width
            b.state[i, j] >= il && push!(v, Int(b.state[i, j]))
            b.state_inv[i, j] >= il && push!(v, Int(b.state_inv[i, j]))
        end
    end
    isempty(v) && return (0, 0, 0, 0.0)
    sort!(v)
    # Fraction strictly above the threshold is the cleanest single number: 0 means every included
    # literal sits exactly on the boundary and there is no gradient at all.
    return (v[1], v[end÷2], v[end], count(>(128), v) / length(v))
end

function arm(budget, seed)
    m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF, budget=budget)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        train!(m, Xtr, Ytr; rng=rng)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
    end
    c = sort(literal_counts(m))
    lo, med, hi, frac = state_spread(m)
    return (best=best, lmed=c[end÷2], lmax=c[end], slo=lo, smed=med, shi=hi, sfrac=frac)
end

arms = (("GrowthGate  [reference]", GrowthGate()),
        ("PartialFreeze  [proposed]", PartialFreeze()),
        ("CapTypeIa   [prior test]", CapTypeIa()))

println("policy                       accuracy          lits med/max   incl-state min/med/max   >thr")
println("-"^88)
results = Dict{String,Vector{Float64}}()
for (name, b) in arms
    accs, info = Float64[], nothing
    for sd in SEEDS
        t = @elapsed (r = arm(b, sd))
        push!(accs, r.best); info = r
    end
    results[name] = accs
    @printf("%-28s %.4f (sd %.4f)  %4d /%4d    %3d /%3d /%3d   %5.1f%%\n",
            name, mean(accs), std(accs), info.lmed, info.lmax,
            info.slo, info.smed, info.shi, 100 * info.sfrac)
end

println()
println("="^88)
base = mean(results["GrowthGate  [reference]"])
for (name, _) in arms
    @printf("%-28s %+.4f vs reference\n", name, mean(results[name]) - base)
end
println()
println("'>thr' is the share of included automata strictly above the include threshold.")
println("Near 0% means every included literal sits on the boundary and there is no gradient to read.")
