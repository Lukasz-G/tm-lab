# Does using the clause vote's magnitude at the feedback boundary help?
#
# FPTM computes an integer clause vote and then discards its magnitude: both Type I and Type II gate
# on `vote > 0`. That is per spec. But measurement says the discarded signal is large — 91.6% of
# nonzero votes on Hnilov's published model, 93.8% on a model trained here, are strictly interior.
# So in about nine firings out of ten, "how well did this clause match" is known and ignored.
#
# ProportionalFeedback treats the vote as a degree of match: Type Ia is applied with probability
# vote/ceiling and Type Ib otherwise, and Type II is accepted with the same probability. It is a
# strict generalisation with no new hyperparameter — the published rule is the step-function limit.
#
# Prediction, stated before running: partial matches now sometimes get eroded where the published
# rule would always reinforce them, so clauses should either sharpen (fewer, more decisive literals,
# accuracy up) or destabilise (clauses churn, accuracy down). A null result is also informative: it
# would mean the discarded magnitude is not actually load-bearing for learning, only for scoring.
#
#   julia --project=. research/proportional-feedback/run.jl [epochs]

include(joinpath(@__DIR__, "..", "mnist.jl"))
using Printf, Random, Statistics
using TMCore

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))
const WIDTH = 784
const CLAUSES, T, S, L, LF = 40, 10, 125, 10, 5
const SEEDS = (20260907, 11, 12, 13, 14, 15, 16, 17, 18, 19)

println("="^78)
println("Vote-proportional feedback vs the published threshold rule")
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

function arm(policy, epochs, seed)
    m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF, feedback=policy)
    rng = MersenneTwister(seed)
    best, final = 0.0, 0.0
    for _ in 1:epochs
        train!(m, Xtr, Ytr; rng=rng)
        final = accuracy(predict(m, Xte), Yte)
        final > best && (best = final)
    end
    c = sort(literal_counts(m))
    # Interior fraction on a test slice: does the change alter how fuzzy the clauses actually are?
    st = observe(m, Xte[1:2000])
    h = vote_histogram(st)
    nz = sum(h[2:end])
    interior = nz == 0 ? 0.0 : (nz - h[end]) / nz
    return (best=best, final=final, med=c[end÷2], max=c[end], interior=interior)
end

# Three arms, because "reinforce less often" and "erode more often" are different edits and the
# obvious formulation makes both at once. ProportionalIdle withholds reinforcement without adding
# erosion, so comparing the two says which half of the change is responsible for any difference.
policies = (("threshold   [published]", ThresholdFeedback()),
            ("proportional", ProportionalFeedback()),
            ("proportional-idle", ProportionalIdle()))

println("policy                    seed        best    final   literals med/max   interior")
println("-"^78)
results = Dict{String,Vector{Float64}}()
for (name, pol) in policies
    accs = Float64[]
    for sd in SEEDS
        t = @elapsed (r = arm(pol, EPOCHS, sd))
        @printf("%-24s  %8d   %.4f   %.4f   %5d /%4d     %.3f   (%.0f s)\n",
                name, sd, r.best, r.final, r.med, r.max, r.interior, t)
        push!(accs, r.best)
    end
    results[name] = accs
end

println()
println("="^78)
base = results["threshold   [published]"]
for name in ("threshold   [published]", "proportional", "proportional-idle")
    a = results[name]
    @printf("%-24s mean %.4f  sd %.5f  (min %.4f, max %.4f)
",
            name, mean(a), std(a), minimum(a), maximum(a))
end

# The arms share seeds, so pair them: the per-seed difference removes seed-to-seed variance and is
# the honest test of a sub-point effect.
println()
for name in ("proportional", "proportional-idle")
    d = results[name] .- base
    se = std(d) / sqrt(length(d))
    @printf("%-20s vs published: mean %+.4f, sd %.5f, se %.5f, worse on %d/%d seeds
",
            name, mean(d), std(d), se, count(<(0), d), length(d))
end

prop, idle = results["proportional"], results["proportional-idle"]
println()
@printf("withholding reinforcement alone : %+.4f
", mean(idle .- base))
@printf("adding erosion on top of it     : %+.4f
", mean(prop .- idle))
