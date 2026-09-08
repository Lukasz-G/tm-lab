# Do fixed edge kernels replicate across datasets?
#
# `booleanization-cifar` measured fixed Sobel/Laplacian kernels at booleanization worth +0.1409 on
# CIFAR-10 with the machine unchanged — the largest single effect anywhere in this repo. Every prior
# positive result here that was tested on a second dataset has broken: core-plus-tail died on IMDb,
# convolutional FPTM reversed on CIFAR-10, learned messages evaporated on CIFAR-10. Three for three.
#
# So this runs the one factor that matters across three datasets before the claim is made, rather
# than after someone else fails to reproduce it. Grayscale everywhere, so the colour factor is out
# and only the kernels move.
#
#   no kernels   [baseline]   thermometer over pixels
#   + kernels                 the same, plus Sobel x, Sobel y and Laplacian responses, pooled
#
# `s = round(width / S)` is held constant across both arms of each dataset, since adding kernel bits
# widens the input and `s` would otherwise change with it — `booleanization` measured that confound
# as larger than the encoder effect it was meant to be studying.
#
# PASS: the kernel arm wins on all three, and the CIFAR-10 margin is not an outlier by an order of
# magnitude. FAIL: it wins on CIFAR-10 only, in which case the effect is a property of CIFAR-10 and
# the write-up says so.
#
#   julia --project=. research/kernel-replication/run.jl [epochs] [seeds]

using Printf, Random, Statistics
using TMCore, TMBoolean

include(joinpath(@__DIR__, "..", "mnist.jl"))
include(joinpath(@__DIR__, "..", "cifar.jl"))

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 25
const NSEEDS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const SEEDS = (20260908, 11, 12, 13, 14)[1:NSEEDS]
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))

const NTRAIN, NTEST = 20_000, 5_000
const CLAUSES, T, L, LF = 40, 10, 10, 5
const S_TARGET = 33          # literals Type Ib erodes per clause; matches booleanization-cifar

matched_S(w) = max(1, round(Int, w / S_TARGET))

"Grayscale pixel matrices in [0,1], `nsamples x (dim*dim)`, plus the image side length."
function load(which::Symbol)
    if which === :cifar
        Xtr, ytr = cifar10(DATA, :train)
        Xte, yte = cifar10(DATA, :test)
        return (cifar_gray(Xtr[1:NTRAIN, :]), Int.(ytr[1:NTRAIN]),
                cifar_gray(Xte[1:NTEST, :]), Int.(yte[1:NTEST]), 32)
    end
    trp, tr_y, n_tr, _, _ = mnist_train(DATA; which=which)
    tep, te_y, n_te, _, _ = mnist_test(DATA; which=which)
    ntr, nte = min(NTRAIN, n_tr), min(NTEST, n_te)
    # mnist_train returns 28x28xN UInt8; flatten row-major per image to match ConvKernel's layout.
    G(px, n) = begin
        M = Matrix{Float64}(undef, n, 784)
        @inbounds for i in 1:n
            M[i, :] = vec(permutedims(view(px, :, :, i))) ./ 255
        end
        M
    end
    return (G(trp, ntr), Int.(tr_y[1:ntr]), G(tep, nte), Int.(te_y[1:nte]), 28)
end

function encode(Atr, Ate, dim, kernels::Bool)
    th = Thermometer(nthresholds=4, strategy=:quantile)
    fit!(th, Atr)
    Btr, Bte = transform(th, Atr), transform(th, Ate)
    if kernels
        ck = ConvKernel(height=dim, width=dim, channels=1, kernels=:edges,
                        nthresholds=2, strategy=:quantile, pool=2, rectify=true)
        fit!(ck, Atr)
        Btr, Bte = hcat(Btr, transform(ck, Atr)), hcat(Bte, transform(ck, Ate))
    end
    return Btr, Bte
end

function run_arm(Btr, Ytr, Bte, Yte, seed)
    w = size(Btr, 2)
    Xtr = [TMInput(Vector{Bool}(view(Btr, i, :))) for i in 1:size(Btr, 1)]
    Xte = [TMInput(Vector{Bool}(view(Bte, i, :))) for i in 1:size(Bte, 1)]
    m = TMClassifier(Ytr, w; clauses_per_class=CLAUSES, T=T, S=matched_S(w), L=L, LF=LF)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        train!(m, Xtr, Ytr; rng=rng)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
    end
    return best
end

println("="^96)
println("Do fixed edge kernels replicate across datasets?")
println("="^96)
@printf("grayscale everywhere, %d clauses/class, T %d, L %d, LF %d, s held at %d\n",
        CLAUSES, T, L, LF, S_TARGET)
@printf("%d train, %d test, %d epochs, %d seeds\n\n", NTRAIN, NTEST, EPOCHS, length(SEEDS))

function main()
    summary = Tuple{Symbol,Float64,Float64,Int,Int}[]
    for which in (:mnist, :fashion, :cifar)
        Atr, Ytr, Ate, Yte, dim = load(which)
        println("-"^96)
        @printf("%s  (%dx%d, %d train, %d test)\n", uppercase(string(which)), dim, dim,
                length(Ytr), length(Yte))
        means = Float64[]
        widths = Int[]
        for kernels in (false, true)
            Btr, Bte = encode(Atr, Ate, dim, kernels)
            w = size(Btr, 2)
            push!(widths, w)
            accs = Float64[]
            label = kernels ? "+ edge kernels" : "no kernels  [baseline]"
            for sd in SEEDS
                t = @elapsed (a = run_arm(Btr, Ytr, Bte, Yte, sd))
                push!(accs, a)
                @printf("  %-24s width %5d  seed %d  %.4f   (%.0f s)\n", label, w, sd, a, t)
            end
            push!(means, mean(accs))
        end
        @printf("  => %+.4f from kernels\n\n", means[2] - means[1])
        push!(summary, (which, means[1], means[2], widths[1], widths[2]))
    end

    println("="^96)
    println("SUMMARY")
    println("="^96)
    @printf("%-10s %10s %10s %10s\n", "dataset", "baseline", "+kernels", "delta")
    for (which, b, k, _, _) in summary
        @printf("%-10s %10.4f %10.4f %+10.4f\n", string(which), b, k, k - b)
    end
    println()
    println("PASS if kernels win on all three. If they win on CIFAR-10 alone, the effect belongs to")
    println("CIFAR-10 and not to booleanization, and the write-up has to say that.")
end

main()
