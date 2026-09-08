using Printf, Random, Statistics
using TMCore

include(joinpath(@__DIR__, "prep.jl"))

const CLAUSES = parse(Int, ARGS[1])
const TVAL = parse(Int, ARGS[2])
const SEED = parse(Int, ARGS[3])
const EPOCHS = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 25

const OUT = joinpath(@__DIR__, "out")
const L, LF = 10, 5
const S_TARGET = 33

# Bits come from the cache rather than from the pixels. The first attempt encoded inside every
# process and seven of twelve died out of memory on a 600 MB Float64 matrix each; this also
# guarantees every point sees byte-identical input.
isfile(joinpath(CACHE, "train.bits")) ||
    error("no cache — run: julia --project=. research/clause-count/prep.jl")
Btr, Ytr = load_bits(joinpath(CACHE, "train.bits"))
Bte, Yte = load_bits(joinpath(CACHE, "test.bits"))
const NTRAIN, NTEST = size(Btr, 1), size(Bte, 1)

const W = size(Btr, 2)
const SVAL = max(1, round(Int, W / S_TARGET))

Xtr = [TMInput(Vector{Bool}(view(Btr, i, :))) for i in 1:NTRAIN]
Xte = [TMInput(Vector{Bool}(view(Bte, i, :))) for i in 1:NTEST]

# Wrapped in a function because a top-level for loop in Julia opens a soft scope and would not
# assign back to best.
function sweep_point()
    m = TMClassifier(Ytr, W; clauses_per_class=CLAUSES, T=TVAL, S=SVAL, L=L, LF=LF)
    rng = MersenneTwister(SEED)
    best, best_epoch = 0.0, 0
    t0 = time()
    for e in 1:EPOCHS
        train!(m, Xtr, Ytr; rng=rng)
        a = accuracy(predict(m, Xte), Yte)
        a > best && ((best, best_epoch) = (a, e))
    end
    return best, best_epoch, time() - t0
end

best, best_epoch, elapsed = sweep_point()

mkpath(OUT)
open(joinpath(OUT, @sprintf("c%04d_T%02d_s%d.txt", CLAUSES, TVAL, SEED)), "w") do io
    # best_epoch is recorded because a peak at the last epoch means the run was still improving and
    # the number understates the arm, which matters most exactly where it is most expensive to fix.
    @printf(io, "%d %d %d %.4f %d %d %.0f\n", CLAUSES, TVAL, SEED, best, best_epoch, EPOCHS, elapsed)
end
@printf("clauses %4d  T %2d  seed %d  acc %.4f  (best at epoch %d of %d, %.0f s)\n",
        CLAUSES, TVAL, SEED, best, best_epoch, EPOCHS, elapsed)
