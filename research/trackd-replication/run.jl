# Do the Track D conclusions survive a second dataset?
#
# Every Track D variant was tested on MNIST alone, and all four lost. The write-ups explained them
# with one mechanism — reducing a clause's tolerance costs it the redundant literals that make it
# robust — which is a satisfying story built on a single dataset at a single hyperparameter setting.
#
# There is direct evidence that such stories are fragile here: the mask-mining interpretability
# result looked clean on MNIST and collapsed on IMDb. So this re-runs the same variants on
# Fashion-MNIST, which shares MNIST's format and shape but is harder, and asks whether the sign and
# rough magnitude of each effect holds.
#
# Same hyperparameters as the MNIST runs, deliberately. They are tuned for MNIST and are probably not
# optimal for Fashion-MNIST — but the question is whether the *variants* behave the same relative to
# a common baseline, not whether the baseline is well tuned. Retuning per dataset would reintroduce
# exactly the confound this is meant to remove.
#
#   julia --project=. research/trackd-replication/run.jl [fashion|mnist|cifar] [epochs]

include(joinpath(@__DIR__, "..", "mnist.jl"))
include(joinpath(@__DIR__, "..", "cifar.jl"))
using Printf, Random, Statistics
using TMCore, TMBoolean
import TMCore: reinforce_allowed, promotion_room

const WHICH = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :fashion
const EPOCHS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))
const CLAUSES, T0, S, L, LF0 = 40, 10, 125, 10, 5
# Seed count is a third argument because a weaker baseline needs more of them: CIFAR-10 sits near
# 0.30 where seed-to-seed spread is far larger than on MNIST, and three seeds cannot resolve a
# half-point effect there.
const NSEEDS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
const SEEDS = (20260908, 11, 12, 13, 14, 15, 16)[1:NSEEDS]

# The clean isolation from budget-paths: reference policy plus a Type II cap and nothing else.
struct GateThenCapII <: TMCore.LiteralBudgetPolicy end
reinforce_allowed(::GateThenCapII, n::Integer, L::Integer) = n <= L
promotion_room(::GateThenCapII, ::Integer, ::Integer, ::TypeIa) = typemax(Int)
promotion_room(::GateThenCapII, n::Integer, L::Integer, ::TypeII) = max(0, Int(L) - Int(n))

println("="^86)
@printf("Track D replication on %s\n", uppercase(String(WHICH)))
println("="^86)
@printf("%d clauses/class, T %d, S %d, L %d, LF %d, %d epochs, %d seeds\n\n",
        CLAUSES, T0, S, L, LF0, EPOCHS, length(SEEDS))

# CIFAR-10 needs its own path: colour, 32x32, and five times wider than MNIST at one bit per channel.
# Grayscale plus a fitted 2-bit thermometer holds the input at 2048 bits so six arms stay affordable.
# Colour is not what is being tested — only whether the variants move the same way.
const CIFAR_TRAIN = 20_000

function load_data()
    if WHICH === :cifar
        Xtr_raw, ytr = cifar10(DATA, :train)
        Xte_raw, yte = cifar10(DATA, :test)
        n = min(CIFAR_TRAIN, size(Xtr_raw, 1))
        Gtr, Gte = cifar_gray(Xtr_raw[1:n, :]), cifar_gray(Xte_raw)
        th = Thermometer(nthresholds=2, strategy=:quantile)
        fit!(th, Gtr)
        Btr, Bte = transform(th, Gtr), transform(th, Gte)
        return ([TMInput(Vector{Bool}(view(Btr, i, :))) for i in 1:size(Btr, 1)], Int.(ytr[1:n]),
                [TMInput(Vector{Bool}(view(Bte, i, :))) for i in 1:size(Bte, 1)], Int.(yte),
                size(Btr, 2))
    end
    tr_px, tr_y, n_tr, _, _ = mnist_train(DATA; which=WHICH)
    te_px, te_y, n_te, _, _ = mnist_test(DATA; which=WHICH)
    tr_bits, _ = booleanize_both(tr_px, n_tr)
    te_bits, _ = booleanize_both(te_px, n_te)
    return ([TMInput(Vector{Bool}(b)) for b in tr_bits], Int.(tr_y),
            [TMInput(Vector{Bool}(b)) for b in te_bits], Int.(te_y), 784)
end

Xtr, Ytr, Xte, Yte, WIDTH_ACTUAL = load_data()
@printf("train %d, test %d, width %d bits\n\n", length(Xtr), length(Xte), WIDTH_ACTUAL)

anneal_down(e, n) = max(1, round(Int, LF0 - (LF0 - 1) * (e - 1) / max(1, n - 1)))

"""
One arm. `mode` selects which single thing differs from the published baseline; everything else is
held at the reference setting so each row is a one-variable change.
"""
function arm(mode, seed, epochs)
    kw = (clauses_per_class=CLAUSES, T=T0, S=S, L=L, LF=LF0)
    m = mode === :proportional ? TMClassifier(Ytr, WIDTH_ACTUAL; kw..., feedback=ProportionalFeedback()) :
        mode === :prop_idle    ? TMClassifier(Ytr, WIDTH_ACTUAL; kw..., feedback=ProportionalIdle()) :
        mode === :weighted     ? TMClassifier(Ytr, WIDTH_ACTUAL; kw..., misscost=ConfidenceWeightedMissCost()) :
        mode === :cap_typeii   ? TMClassifier(Ytr, WIDTH_ACTUAL; kw..., budget=GateThenCapII()) :
                                 TMClassifier(Ytr, WIDTH_ACTUAL; kw...)
    rng = MersenneTwister(seed)
    best, final = 0.0, 0.0
    for e in 1:epochs
        if mode === :anneal_lf
            lf = anneal_down(e, epochs)
            set_hyper!(m; LF=lf, T=max(1, round(Int, T0 * sqrt(lf / LF0))))
        elseif mode === :weighted
            calibrate_misscost!(m)     # threshold must track the state distribution as it moves
        end
        train!(m, Xtr, Ytr; rng=rng)
        final = accuracy(predict(m, Xte), Yte)
        final > best && (best = final)
    end
    c = sort(literal_counts(m))
    return (best=best, final=final, med=c[end÷2], max=c[end])
end

modes = ((:baseline,     "baseline  [published]"),
         (:proportional, "proportional feedback"),
         (:prop_idle,    "proportional-idle"),
         (:weighted,     "confidence-weighted miss cost"),
         (:anneal_lf,    "anneal LF 5->1, T rescaled"),
         (:cap_typeii,   "L gates Type II too"))

println("variant                          best (per seed)              mean    final   lits med/max")
println("-"^86)
results = Dict{Symbol,Vector{Float64}}()
for (mode, name) in modes
    bs, fs, info = Float64[], Float64[], nothing
    for sd in SEEDS
        t = @elapsed (r = arm(mode, sd, EPOCHS))
        push!(bs, r.best); push!(fs, r.final); info = r
    end
    results[mode] = bs
    @printf("%-30s %s   %.4f (sd %.4f)  %.4f   %4d /%4d\n", name,
            join([@sprintf("%.4f", b) for b in bs], " "), mean(bs), std(bs), mean(fs),
            info.med, info.max)
end

println()
println("="^86)
base = results[:baseline]
@printf("%-30s %s\n", "", "vs baseline    worse on")
for (mode, name) in modes
    mode === :baseline && continue
    d = results[mode] .- base
    se = std(d) / sqrt(length(d))
    @printf("%-30s   %+.4f (se %.4f)   %d/%d seeds%s\n", name, mean(d), se,
            count(<(0), d), length(d), abs(mean(d)) < 2se ? "   <- inside the noise" : "")
end
println()
println("MNIST reference, for comparison (30 epochs, from the original runs):")
println("  proportional -0.0042 | prop-idle -0.0040 | weighted -0.0028 | anneal -0.0050 | capII -0.1339")
