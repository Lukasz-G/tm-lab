# Encode CIFAR-10 once and cache the bits, so sweep processes never touch the pixel data.
#
# The first attempt at the sweep launched twelve processes that each called `cifar_rgb`, which
# materialises a 20000x3072 Float64 matrix — about 600 MB — before any of it can be freed. Seven of
# the twelve died out of memory during that peak. Encoding once and caching removes the peak
# entirely, and has the side benefit that every point of the sweep provably sees identical bits
# rather than identically-constructed ones.
#
# The format is deliberately trivial and matches the project's stance on serialization: a small
# header of Int64 dimensions followed by the raw packed chunks, no Julia `Serialization`.
#
#   julia --project=. research/clause-count/prep.jl

using TMBoolean

include(joinpath(@__DIR__, "..", "cifar.jl"))

const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))
const CACHE = joinpath(@__DIR__, "cache")
const NTRAIN, NTEST = 20_000, 5_000
const DIM = 32

function save_bits(path, B::BitMatrix, y::Vector{Int})
    open(path, "w") do io
        write(io, Int64(size(B, 1)), Int64(size(B, 2)))
        write(io, B.chunks)
        write(io, Int64.(y))
    end
end

function load_bits(path)
    open(path, "r") do io
        n = read(io, Int64); m = read(io, Int64)
        B = falses(n, m)
        read!(io, B.chunks)
        y = Vector{Int64}(undef, n)
        read!(io, y)
        return B, Int.(y)
    end
end

function main()
    mkpath(CACHE)
    Xtr_raw, ytr = cifar10(DATA, :train)
    Xte_raw, yte = cifar10(DATA, :test)
    Rtr, Rte = cifar_rgb(Xtr_raw[1:NTRAIN, :]), cifar_rgb(Xte_raw[1:NTEST, :])

    # The winning encoder from `booleanization-cifar`, unchanged, so the 40-clause point of the
    # sweep reproduces that experiment rather than merely resembling it.
    th = Thermometer(nthresholds=2, strategy=:quantile)
    fit!(th, Rtr)
    ck = ConvKernel(height=DIM, width=DIM, channels=3, kernels=:edges,
                    nthresholds=2, strategy=:quantile, pool=2, rectify=true)
    fit!(ck, Rtr)

    Btr = hcat(transform(th, Rtr), transform(ck, Rtr))
    Bte = hcat(transform(th, Rte), transform(ck, Rte))
    save_bits(joinpath(CACHE, "train.bits"), Btr, Int.(ytr[1:NTRAIN]))
    save_bits(joinpath(CACHE, "test.bits"), Bte, Int.(yte[1:NTEST]))
    println("cached ", size(Btr), " train and ", size(Bte), " test bits in ", CACHE)
end

# Guarded so that run.jl can include this file for load_bits without re-encoding.
if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
