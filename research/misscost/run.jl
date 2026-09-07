# Does charging a failed literal by its automaton's confidence help?
#
# FPTM charges exactly 1 per failed literal. The model already knows how strongly it believes each
# literal belongs — an automaton just over the include threshold is a tentative inclusion, one near
# state_max a confident one — and evaluation discards that. Charging confident misses double is the
# obvious use of it, and costs no new hyperparameter.
#
# The obvious threshold is the midpoint of the nominal include band. It is also useless, and finding
# out why is most of what this experiment produced: under reference training, included automata never
# get anywhere near it. So the arms below include a *calibrated* threshold — the median of the
# model's own included-automaton states — to test the idea rather than a threshold guess.
#
#   julia --project=. research/misscost/run.jl [epochs]

include(joinpath(@__DIR__, "..", "mnist.jl"))
using Printf, Random, Statistics
using TMCore

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))
const WIDTH = 784
const CLAUSES, T, S, L, LF = 40, 10, 125, 10, 5
const SEEDS = (20260907, 11, 12, 13, 14)

println("="^78)
println("Confidence-weighted miss cost vs the published uniform cost")
println("="^78)
@printf("MNIST, %d clauses/class, T %d, S %d, L %d, LF %d, %d epochs, %d seeds\n\n",
        CLAUSES, T, S, L, LF, EPOCHS, length(SEEDS))

tr_px, tr_y, n_tr, _, _ = mnist_train(DATA)
te_px, te_y, n_te, _, _ = mnist_test(DATA)
tr_bits, _ = booleanize_both(tr_px, n_tr)
te_bits, _ = booleanize_both(te_px, n_te)
Xtr = [TMInput(Vector{Bool}(b)) for b in tr_bits]
Xte = [TMInput(Vector{Bool}(b)) for b in te_bits]
Ytr, Yte = Int.(tr_y), Int.(te_y)

"Distribution of included automaton states — the quantity the whole idea depends on existing."
function state_spread(m)
    v = UInt16[]
    for banks in (m.positive, m.negative), b in banks
        for j in 1:b.nclauses, i in 1:b.width
            b.state[i, j] >= b.include_limit && push!(v, UInt16(b.state[i, j]))
            b.state_inv[i, j] >= b.include_limit && push!(v, UInt16(b.state_inv[i, j]))
        end
    end
    isempty(v) && return (0, 0, 0)
    sort!(v)
    return (Int(v[1]), Int(v[end÷2]), Int(v[end]))
end

function arm(mode, epochs, seed)
    mc = mode === :uniform ? UniformMissCost() : ConfidenceWeightedMissCost()
    m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF, misscost=mc)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:epochs
        # The calibrated arm re-fits its threshold each epoch, because the state distribution moves
        # as training proceeds and a stale threshold would drift back towards being a no-op.
        mode === :calibrated && calibrate_misscost!(m)
        train!(m, Xtr, Ytr; rng=rng)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
    end
    lo, med, hi = state_spread(m)
    thr = m.misscost isa ConfidenceWeightedMissCost ?
          Int(confidence_threshold(m.misscost, m.positive[1])) : 0
    return (best=best, lo=lo, med=med, hi=hi, thr=thr)
end

modes = ((:uniform, "uniform  [published]"),
         (:nominal, "weighted, nominal midpoint"),
         (:calibrated, "weighted, calibrated median"))

println("policy                          seed      best   incl-state min/med/max   threshold")
println("-"^78)
results = Dict{Symbol,Vector{Float64}}()
for (mode, name) in modes
    accs = Float64[]
    for sd in SEEDS
        t = @elapsed (r = arm(mode, EPOCHS, sd))
        @printf("%-30s %8d  %.4f      %3d /%3d /%3d        %4d   (%.0f s)\n",
                name, sd, r.best, r.lo, r.med, r.hi, r.thr, t)
        push!(accs, r.best)
    end
    results[mode] = accs
end

println()
println("="^78)
base = results[:uniform]
for (mode, name) in modes
    a = results[mode]
    d = a .- base
    @printf("%-30s mean %.4f   %+.4f vs published   worse on %d/%d seeds\n",
            name, mean(a), mean(d), count(<(0), d), length(d))
end
