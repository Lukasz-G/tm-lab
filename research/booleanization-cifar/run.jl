# Track C on CIFAR-10: how much of the gap to published numbers is the encoder?
#
# `cifar-messages` measured flat FPTM at 0.3528 on CIFAR-10 under grayscale with four thermometer
# thresholds. Published convolutional-TM figures on this dataset are in the 60s. If a large share of
# reported TM accuracy really lives in booleanization rather than in the machine — which is the whole
# premise of this track, and which FPTM's own Fashion-MNIST result leans on by using fixed
# convolutional kernels the paper disclaims as not part of the TM — then changing only the encoder
# should move that number a long way.
#
# DESIGN. A 2x2 factorial rather than a list of encoders, so the two factors can be read separately
# and their interaction is visible:
#
#                     no kernels          + fixed edge kernels
#     grayscale       baseline            kernels alone
#     RGB             colour alone        both
#
# The machine is identical in every arm: flat FPTM, no patches, no messages. Only the encoder moves.
#
# THE CONFOUND THIS CONTROLS. `s = round(width / S)` is the number of literals Type Ib erodes per
# clause, so it scales with input width. An encoder that widens the input silently weakens forgetting
# unless `S` is rescaled with it — `booleanization` measured that effect as larger than the encoder
# effect it was supposed to be studying. Every arm here therefore holds `s` fixed at the baseline's
# value and lets `S` vary. Without that, this table would measure `s`.
#
#   julia --project=. research/booleanization-cifar/run.jl [epochs] [seeds]

using Printf, Random, Statistics
using TMCore, TMBoolean

include(joinpath(@__DIR__, "..", "cifar.jl"))

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 25
const NSEEDS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const SEEDS = (20260908, 11, 12, 13, 14)[1:NSEEDS]
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))

const NTRAIN, NTEST = 20_000, 5_000
const DIM = 32
const CLAUSES, T, L, LF = 40, 10, 10, 5

# The baseline is `cifar-messages`' encoder, so its number is directly comparable: grayscale, four
# thermometer thresholds, S = 125 over 4096 bits.
const BASE_WIDTH, BASE_S = 4096, 125
const S_TARGET = round(Int, BASE_WIDTH / BASE_S)     # literals Type Ib erodes per clause: 33

"`S` that reproduces the baseline's `s` at a different width. This is the whole width control."
matched_S(w) = max(1, round(Int, w / S_TARGET))

"The s that actually results, since S must be an integer and the division rarely lands exactly."
actual_s(w) = max(1, round(Int, w / matched_S(w)))

println("="^100)
println("Track C on CIFAR-10: colour and fixed kernels, at constant s")
println("="^100)
@printf("flat FPTM in every arm: %d clauses/class, T %d, L %d, LF %d, %d train, %d test, %d epochs, %d seeds\n",
        CLAUSES, T, L, LF, NTRAIN, NTEST, EPOCHS, length(SEEDS))
@printf("s held at %d in every arm by setting S = width / %d\n\n", S_TARGET, S_TARGET)

print("loading CIFAR-10 ... ")
Xtr_raw, ytr_all = cifar10(DATA, :train)
Xte_raw, yte_all = cifar10(DATA, :test)
const GTR, GTE = cifar_gray(Xtr_raw[1:NTRAIN, :]), cifar_gray(Xte_raw[1:NTEST, :])
const RTR, RTE = cifar_rgb(Xtr_raw[1:NTRAIN, :]), cifar_rgb(Xte_raw[1:NTEST, :])
const YTR, YTE = Int.(ytr_all[1:NTRAIN]), Int.(yte_all[1:NTEST])
println("done")

"""
Build the bit matrices for one arm.

Encoders are fitted on train and applied to train and test separately — never `fit_transform!` on
both — because thermometer thresholds fitted over the test set are lookahead, and on an image
benchmark it would inflate every arm silently and unequally.
"""
function encode(colour::Bool, nth::Int, kernels::Bool)
    Atr, Ate = colour ? (RTR, RTE) : (GTR, GTE)
    nch = colour ? 3 : 1
    parts_tr, parts_te, names = Any[], Any[], String[]

    th = Thermometer(nthresholds=nth, strategy=:quantile)
    fit!(th, Atr)
    push!(parts_tr, transform(th, Atr)); push!(parts_te, transform(th, Ate))
    append!(names, feature_names(th))

    if kernels
        ck = ConvKernel(height=DIM, width=DIM, channels=nch, kernels=:edges,
                        nthresholds=2, strategy=:quantile, pool=2, rectify=true)
        fit!(ck, Atr)
        push!(parts_tr, transform(ck, Atr)); push!(parts_te, transform(ck, Ate))
        append!(names, feature_names(ck))
    end

    Btr = length(parts_tr) == 1 ? parts_tr[1] : hcat(parts_tr...)
    Bte = length(parts_te) == 1 ? parts_te[1] : hcat(parts_te...)
    return Btr, Bte, names
end

function run_arm(Btr, Bte, seed)
    w = size(Btr, 2)
    Xtr = [TMInput(Vector{Bool}(view(Btr, i, :))) for i in 1:size(Btr, 1)]
    Xte = [TMInput(Vector{Bool}(view(Bte, i, :))) for i in 1:size(Bte, 1)]
    m = TMClassifier(YTR, w; clauses_per_class=CLAUSES, T=T, S=matched_S(w), L=L, LF=LF)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        train!(m, Xtr, YTR; rng=rng)
        a = accuracy(predict(m, Xte), YTE)
        a > best && (best = a)
    end
    return best
end

# RGB is encoded at 2 thresholds and grayscale at 4, so "RGB, no kernels" differs from the baseline
# in colour *and* in per-pixel resolution *and* in width. Grayscale at 6 thresholds is 6144 bits, the
# same width as RGB at 2 and at the same s, which is the arm that isolates colour from the other two.
# Without it the colour row measures three things at once.
const ARMS = (("grayscale x4, no kernels [baseline]", false, 4, false),
              ("grayscale x6, no kernels [width ctl]", false, 6, false),
              ("RGB x2, no kernels", true, 2, false),
              ("grayscale x4 + edge kernels", false, 4, true),
              ("RGB x2 + edge kernels", true, 2, true))

function main()
    res = Dict{String,Vector{Float64}}()
    widths = Dict{String,Int}()
    println("arm                                  accuracy per seed")
    println("-"^100)
    for (name, colour, nth, kernels) in ARMS
        Btr, Bte, names = encode(colour, nth, kernels)
        w = size(Btr, 2)
        widths[name] = w
        # A bit that is never set, or always set, costs a literal and carries nothing. Reporting it
        # per arm keeps a width comparison honest: two encoders of equal width are not equal if one
        # of them is half dead.
        colmean = vec(mean(Btr, dims=1))
        dead = count(x -> x < 1e-9 || x > 1 - 1e-9, colmean)
        @printf("  %-36s width %5d, s %2d, %d dead bits\n",
                name, w, actual_s(w), dead)
        accs = Float64[]
        for sd in SEEDS
            t = @elapsed (a = run_arm(Btr, Bte, sd))
            push!(accs, a)
            @printf("  %-36s seed %d  %.4f   (%.0f s)\n", name, sd, a, t)
        end
        res[name] = accs
        length(names) == w || error("feature_names disagrees with width for $name")
    end

    println()
    println("="^100)
    base = mean(res[ARMS[1][1]])
    for (name, _, _, _) in ARMS
        @printf("%-36s mean %.4f   %+.4f vs baseline   (width %d)\n",
                name, mean(res[name]), mean(res[name]) - base, widths[name])
    end
    println()
    println("Read RGB against the grayscale x6 width control, not against the baseline: that pair")
    println("differs only in colour. Baseline-relative numbers fold in width and threshold count too.")
end

main()
