# Does annealing LF from fuzzy toward strict help?
#
# The idea: fuzziness early gives a gradient-like signal that a strict all-or-nothing clause cannot,
# so a model should learn faster with high LF; sharpening toward LF = 1 late should then recover the
# precision of a strict TM. Fuzzy for the search, strict for the finish.
#
# There is a confound to control. A clause's maximum vote is its ceiling, which LF bounds, so
# lowering LF shrinks the entire vote scale — while T, the threshold that scale is compared against,
# stays put. arXiv:2508.08350 §2.1 puts the optimum near sqrt(CLAUSES/2 * LF), so annealing LF alone
# walks steadily away from the recommended relation. An arm that rescales T alongside separates
# "annealing is bad" from "annealing broke the T calibration".
#
# Prediction, stated before running. proportional-feedback showed that redundant literals are what
# make these models robust, and LF = 1 removes the tolerance that makes redundancy useful — so the
# late-strict phase should hurt. Whether the early-fuzzy phase buys enough to compensate is the open
# part. A reversed arm (strict early, fuzzy late) is included as a direction check: if annealing
# down and annealing up both land in the same place, the schedule is doing nothing and only the
# endpoint matters.
#
#   julia --project=. research/annealed-lf/run.jl [epochs]

include(joinpath(@__DIR__, "..", "mnist.jl"))
using Printf, Random, Statistics
using TMCore

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))
const WIDTH = 784
const CLAUSES, T0, S, L, LF0 = 40, 10, 125, 10, 5
const SEEDS = (20260907, 11, 12, 13, 14)

println("="^80)
println("Annealed LF vs the published constant LF")
println("="^80)
@printf("MNIST, %d clauses/class, T %d, S %d, L %d, LF %d, %d epochs, %d seeds\n\n",
        CLAUSES, T0, S, L, LF0, EPOCHS, length(SEEDS))

tr_px, tr_y, n_tr, _, _ = mnist_train(DATA)
te_px, te_y, n_te, _, _ = mnist_test(DATA)
tr_bits, _ = booleanize_both(tr_px, n_tr)
te_bits, _ = booleanize_both(te_px, n_te)
Xtr = [TMInput(Vector{Bool}(b)) for b in tr_bits]
Xte = [TMInput(Vector{Bool}(b)) for b in te_bits]
Ytr, Yte = Int.(tr_y), Int.(te_y)

# Schedules: epoch index and total -> LF for that epoch.
constant(_, _) = LF0
anneal_down(e, n) = max(1, round(Int, LF0 - (LF0 - 1) * (e - 1) / max(1, n - 1)))
anneal_up(e, n) = max(1, round(Int, 1 + (LF0 - 1) * (e - 1) / max(1, n - 1)))

function arm(schedule, rescale_T, epochs, seed)
    m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T0, S=S, L=L, LF=LF0)
    rng = MersenneTwister(seed)
    best, trace = 0.0, Int[]
    for e in 1:epochs
        lf = schedule(e, epochs)
        # Keep T on the paper's relation when asked: T scales as sqrt(LF).
        t = rescale_T ? max(1, round(Int, T0 * sqrt(lf / LF0))) : T0
        set_hyper!(m; LF=lf, T=t)
        push!(trace, lf)
        train!(m, Xtr, Ytr; rng=rng)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
    end
    final = accuracy(predict(m, Xte), Yte)
    c = sort(literal_counts(m))
    return (best=best, final=final, med=c[end÷2], max=c[end], lf_end=trace[end])
end

arms = (("constant LF=5  [published]", constant, false),
        ("anneal 5->1, T fixed", anneal_down, false),
        ("anneal 5->1, T rescaled", anneal_down, true),
        ("anneal 1->5, T fixed  [rev]", anneal_up, false))

println("arm                            seed      best    final   literals med/max   LF_end")
println("-"^80)
results = Dict{String,Vector{Float64}}()
finals = Dict{String,Vector{Float64}}()
for (name, sched, rescale) in arms
    bs, fs = Float64[], Float64[]
    for sd in SEEDS
        t = @elapsed (r = arm(sched, rescale, EPOCHS, sd))
        @printf("%-30s %8d  %.4f   %.4f     %4d /%4d      %d   (%.0f s)\n",
                name, sd, r.best, r.final, r.med, r.max, r.lf_end, t)
        push!(bs, r.best); push!(fs, r.final)
    end
    results[name] = bs
    finals[name] = fs
end

println()
println("="^80)
base = results["constant LF=5  [published]"]
for (name, _, _) in arms
    a = results[name]
    d = a .- base
    @printf("%-30s best mean %.4f  %+.4f   worse on %d/%d   (final mean %.4f)\n",
            name, mean(a), mean(d), count(<(0), d), length(d), mean(finals[name]))
end
