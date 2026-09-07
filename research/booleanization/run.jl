# How much of a TM's accuracy lives in the booleanizer rather than in the machine?
#
# The design notes claim much of the reported TM gain sits here: FPTM's Fashion-MNIST
# state-of-the-art leans on fixed convolutional kernels applied at booleanization, which the paper
# explicitly disclaims as not part of the TM. If that is right, then comparisons between TM variants
# that also vary the encoder are not comparisons between TMs, and the encoder deserves to be a named,
# measured component rather than a preprocessing footnote.
#
# Concretely: FuzzyPatternTM's own MNIST files disagree with each other. The training example encodes
# at four thresholds, `booleanize(x, 0, 0.25, 0.5, 0.75)`; the inference benchmark encodes at one,
# `booleanize(x, 0.25)`. Same machine, same hyperparameters, 4x the input width. Nobody reports what
# that is worth.
#
# THE CONFOUND, and it is not optional. `s`, the number of automata Type Ib erodes per event, is
# derived as `width / S`. Widening the input from 784 to 3136 bits quadruples `s` unless `S` is
# scaled with it — so a naive comparison of 1-bit against 4-bit changes the forgetting rate fourfold
# and attributes the result to the encoder. Arms below hold `s` fixed by scaling `S`, and one control
# deliberately does not, to show how much that alone is worth.
#
#   julia --project=. research/booleanization/run.jl [epochs]

include(joinpath(@__DIR__, "..", "mnist.jl"))
using Printf, Random, Statistics
using TMCore, TMBoolean

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))
const CLAUSES, T, S0, L, LF = 40, 10, 125, 10, 5
const W0 = 784                      # baseline width, and the width S0 was tuned at
const SEEDS = (20260908, 11, 12)

println("="^84)
println("How much accuracy lives in the booleanizer?")
println("="^84)
@printf("MNIST, %d clauses/class, T %d, L %d, LF %d, %d epochs, %d seeds\n\n",
        CLAUSES, T, L, LF, EPOCHS, length(SEEDS))

tr_px, tr_y, n_tr, _, _ = mnist_train(DATA)
te_px, te_y, n_te, _, _ = mnist_test(DATA)

"Raw pixels as nsamples x 784 Float64 in [0,1], in the orientation the upstream models use."
function pixel_matrix(px, n)
    M = Matrix{Float64}(undef, n, 784)
    @inbounds for i in 1:n
        v = vec(view(px, :, :, i))
        for j in 1:784
            M[i, j] = v[j] / 255
        end
    end
    return M
end

Mtr, Mte = pixel_matrix(tr_px, n_tr), pixel_matrix(te_px, n_te)
Ytr, Yte = Int.(tr_y), Int.(te_y)
@printf("train %d, test %d, %d raw pixels\n\n", n_tr, n_te, size(Mtr, 2))

"Fixed thresholds on the [0,1] scale, exactly as upstream's `booleanize(x, ts...)` does."
function fixed_encode(M, ts)
    n, f = size(M)
    B = falses(n, f * length(ts))
    @inbounds for j in 1:f, (c, t) in enumerate(ts)
        col = (j - 1) * length(ts) + c
        for i in 1:n
            B[i, col] = M[i, j] > t
        end
    end
    return B
end

to_inputs(B) = [TMInput(Vector{Bool}(view(B, i, :))) for i in 1:size(B, 1)]

function run_arm(name, Btr, Bte, scale_S, seed)
    w = size(Btr, 2)
    # Hold s = width/S constant unless the arm is the control that deliberately does not.
    S = scale_S ? max(1, round(Int, S0 * w / W0)) : S0
    Xtr, Xte = to_inputs(Btr), to_inputs(Bte)
    m = TMClassifier(Ytr, w; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        train!(m, Xtr, Ytr; rng=rng)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
    end
    return (best=best, width=w, S=S, s=m.params.s)
end

# Encoders. The two "fixed" arms are upstream's own two choices; the thermometer arms fit thresholds
# to the training data instead of assuming them, which is the thing a library can offer that a
# hardcoded call cannot.
therm4 = Thermometer(nthresholds=4, strategy=:quantile)
fit!(therm4, Mtr)
therm2 = Thermometer(nthresholds=2, strategy=:quantile)
fit!(therm2, Mtr)

arms = (
    ("fixed 1 bit  (0.25)",        fixed_encode(Mtr, (0.25,)),                 fixed_encode(Mte, (0.25,)),                 true),
    ("fixed 4 bit  (0,.25,.5,.75)", fixed_encode(Mtr, (0.0, 0.25, 0.5, 0.75)), fixed_encode(Mte, (0.0, 0.25, 0.5, 0.75)), true),
    ("thermo 2 bit (quantile)",    transform(therm2, Mtr),                     transform(therm2, Mte),                     true),
    ("thermo 4 bit (quantile)",    transform(therm4, Mtr),                     transform(therm4, Mte),                     true),
    ("fixed 4 bit, S NOT scaled",  fixed_encode(Mtr, (0.0, 0.25, 0.5, 0.75)), fixed_encode(Mte, (0.0, 0.25, 0.5, 0.75)), false),
)

println("encoder                      width     S    s     best accuracy (per seed)      mean")
println("-"^84)
results = Dict{String,Vector{Float64}}()
for (name, Btr, Bte, scale) in arms
    accs = Float64[]
    info = nothing
    for sd in SEEDS
        t = @elapsed (r = run_arm(name, Btr, Bte, scale, sd))
        push!(accs, r.best)
        info = r
    end
    results[name] = accs
    @printf("%-28s %5d %5d %4d    %s   %.4f\n", name, info.width, info.S, info.s,
            join([@sprintf("%.4f", a) for a in accs], " "), mean(accs))
end

println()
println("="^84)
base = mean(results["fixed 1 bit  (0.25)"])
for (name, _, _, _) in arms
    @printf("%-28s %+.4f vs the 1-bit baseline\n", name, mean(results[name]) - base)
end
println()
@printf("encoder alone (1 -> 4 fixed bits, s held): %+.4f\n",
        mean(results["fixed 4 bit  (0,.25,.5,.75)"]) - base)
@printf("fitted vs assumed thresholds at 4 bits   : %+.4f\n",
        mean(results["thermo 4 bit (quantile)"]) - mean(results["fixed 4 bit  (0,.25,.5,.75)"]))
@printf("cost of not rescaling S with width       : %+.4f\n",
        mean(results["fixed 4 bit, S NOT scaled"]) - mean(results["fixed 4 bit  (0,.25,.5,.75)"]))
