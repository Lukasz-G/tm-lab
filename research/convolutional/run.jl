# Stage 3: per-node evaluation, zero message rounds.
#
# A convolutional FPTM. An image is cut into patches; each patch is a "node"; a clause evaluates on
# every patch independently and the patch outputs combine into one clause output for the image. No
# messages yet, so this is Track E's first step with the graph part removed.
#
# Gate, from the plan: beat flat FPTM on identical booleanization at the same clause count. The flat
# baseline runs here rather than being quoted, so the comparison is like-for-like.
#
# Two of the four undecided semantics have to be settled to write this at all, and neither has an
# answer in either paper. They are the arms:
#
#   FUZZY OR ACROSS NODES — how patch votes become the clause's output.
#       max   the natural analogue of OR: "this pattern appears somewhere"
#       sum   a counting model: "this pattern appears often", whose scale depends on patch count and
#             therefore interacts with T
#
#   CREDIT ASSIGNMENT — which patch feedback acts on.
#       random   uniform over patches where the clause fired; what GraphTM's CUDA and the classical
#                convolutional TM both do
#       argmax   the best-scoring patch. Under fuzzy evaluation every patch scores nonzero, so the
#                reference rule is arguably no longer the natural one
#
# Position is thermometer-encoded and appended to each patch, so a clause can be position-sensitive
# or translation-invariant as it prefers.
#
#   julia --project=. research/convolutional/run.jl [epochs] [seeds] [clauses] [stride]

include(joinpath(@__DIR__, "..", "mnist.jl"))
using Printf, Random, Statistics
using TMCore
import TMCore: reinforce_allowed, promotion_room

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 15
const NSEEDS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const SEEDS = (20260908, 11, 12, 13, 14)[1:NSEEDS]
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))
const NTRAIN = 10_000

const DIM, PATCH = 28, 10
# Stride is an argument because it is the most likely way to cripple a convolutional model by
# accident: a coarse stride gives few patch positions, and matching a pattern "somewhere" is worth
# less when there are only 25 somewheres. Classical convolutional TMs use stride 1.
const STRIDE = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 4
# Clause count is an argument because the real argument for convolution is clause *efficiency*: a
# patch pattern is reusable across positions, so conv should need fewer clauses for the same
# accuracy. A single clause count cannot show that either way.
const CLAUSES = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 40
const T, S, L, LF = 10, 125, 10, 5

const POS = 0:STRIDE:(DIM - PATCH)
const NPOS = length(POS)
const PATCH_BITS = PATCH * PATCH
const WIDTH = PATCH_BITS + 2 * (NPOS - 1)
const NPATCH = NPOS * NPOS

println("="^92)
println("Stage 3 — convolutional FPTM: per-node evaluation, zero message rounds")
println("="^92)
@printf("MNIST %dx%d, patch %d, stride %d -> %d patches of %d bits (%d pixel + %d position)\n",
        DIM, DIM, PATCH, STRIDE, NPATCH, WIDTH, PATCH_BITS, 2 * (NPOS - 1))
@printf("%d clauses/class, T %d, S %d, L %d, LF %d, %d train, %d epochs, %d seeds\n\n",
        CLAUSES, T, S, L, LF, NTRAIN, EPOCHS, length(SEEDS))

"""
Patches of one booleanized image, each with its position appended thermometer-style.

Thermometer rather than one-hot for the usual reason: the bits nest, so a clause can say "row >= 2"
with a single literal instead of needing a disjunction it cannot express.
"""
function patches_of(img::BitVector)
    out = Vector{TMInput}(undef, NPATCH)
    k = 0
    for (yi, y) in enumerate(POS), (xi, x) in enumerate(POS)
        v = falses(WIDTH)
        @inbounds for r in 0:(PATCH - 1), c in 0:(PATCH - 1)
            v[r * PATCH + c + 1] = img[(y + r) * DIM + (x + c) + 1]
        end
        off = PATCH_BITS
        @inbounds for i in 1:(NPOS - 1)
            v[off + i] = yi > i
            v[off + (NPOS - 1) + i] = xi > i
        end
        k += 1
        out[k] = TMInput(v)
    end
    return out
end

trp, tryy, ntr, _, _ = mnist_train(DATA)
tep, tey, nte, _, _ = mnist_test(DATA)
trb, _ = booleanize_both(trp, min(NTRAIN, ntr))
teb, _ = booleanize_both(tep, nte)
Ytr, Yte = Int.(tryy[1:length(trb)]), Int.(tey)

print("building patches ... ")
Ptr = [patches_of(b) for b in trb]
Pte = [patches_of(b) for b in teb]
Xtr_flat = [TMInput(Vector{Bool}(b)) for b in trb]
Xte_flat = [TMInput(Vector{Bool}(b)) for b in teb]
println("done")

"""
    clause_over_patches(bank, j, patches, LF, ceilpol, agg, credit, rng) -> (value, patch)

The clause's output for a whole image, plus the patch feedback should act on. Patch `0` means no
patch fired, which routes the clause to Type Ib.
"""
@inline function clause_over_patches(b::ClauseBank, j::Integer, patches, LF, ceilpol,
                                     agg::Symbol, credit::Symbol, rng)
    best, chosen, total, nfired = 0, 0, 0, 0
    @inbounds for k in eachindex(patches)
        v = clause_vote(b, j, patches[k], LF, ceilpol)
        v == 0 && continue
        nfired += 1
        total += v
        if credit === :random
            # Reservoir sample of size one: uniform over firing patches in a single pass, without
            # materialising the list. GraphTM's CUDA builds that list only because a per-thread
            # fixed array forces it to; nothing here does.
            rand(rng) * nfired < 1 && (chosen = k)
        elseif v > best
            chosen = k
        end
        v > best && (best = v)
    end
    nfired == 0 && return (0, 0)
    return (agg === :sum ? total : best, chosen)
end

function conv_score(m, ci, patches, agg)
    LF, cp = m.params.LF, m.ceiling
    pos = neg = 0
    @inbounds for j in 1:m.positive[ci].nclauses
        v, _ = clause_over_patches(m.positive[ci], j, patches, LF, cp, agg, :argmax, nothing)
        pos += v
    end
    @inbounds for j in 1:m.negative[ci].nclauses
        v, _ = clause_over_patches(m.negative[ci], j, patches, LF, cp, agg, :argmax, nothing)
        neg += v
    end
    return pos - neg
end

function conv_predict(m, patches, agg)
    best_i, best_v = 1, typemin(Int)
    for ci in eachindex(m.classes)
        v = conv_score(m, ci, patches, agg)
        v > best_v && ((best_v, best_i) = (v, ci))
    end
    return m.classes[best_i]
end

function conv_update!(m, ci, patches, positive, rng, agg, credit)
    p = m.params
    Tt, LFv, Lv, sv, cp = p.T, p.LF, p.L, p.s, m.ceiling
    v = clamp(conv_score(m, ci, patches, agg), -Tt, Tt)
    upd = (positive ? (Tt - v) : (Tt + v)) / (2Tt)
    tI  = positive ? m.positive[ci] : m.negative[ci]
    tII = positive ? m.negative[ci] : m.positive[ci]
    @inbounds for j in 1:tI.nclauses
        rand(rng) < upd || continue
        _, k = clause_over_patches(tI, j, patches, LFv, cp, agg, credit, rng)
        n = Int(tI.count[j])
        if k != 0
            feedback!(TypeIa(), tI, j, patches[k], reinforce_allowed(m.budget, n, Lv),
                      promotion_room(m.budget, n, Lv, TypeIa()))
        else
            feedback!(TypeIb(), tI, j, sv, rng)
        end
    end
    @inbounds for j in 1:tII.nclauses
        rand(rng) < upd || continue
        _, k = clause_over_patches(tII, j, patches, LFv, cp, agg, credit, rng)
        k == 0 && continue
        feedback!(TypeII(), tII, j, patches[k],
                  promotion_room(m.budget, Int(tII.count[j]), Lv, TypeII()))
    end
end

function conv_train!(m, P, Y, rng, agg, credit)
    for i in randperm(rng, length(Y))
        yi = findfirst(==(Y[i]), m.classes)
        conv_update!(m, yi, P[i], true, rng, agg, credit)
        for ci in eachindex(m.classes)
            ci == yi || conv_update!(m, ci, P[i], false, rng, agg, credit)
        end
    end
end

function run_flat(seed)
    m = TMClassifier(Ytr, 784; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        train!(m, Xtr_flat, Ytr; rng=rng)
        a = accuracy(predict(m, Xte_flat), Yte)
        a > best && (best = a)
    end
    return best
end

# s, the Type Ib erosion count, is derived as width/S. A patch is 136 bits against the flat model's
# 784, so an unscaled comparison silently gives the convolutional arm a sixth of the forgetting rate
# — exactly the confound `booleanization/` found dominating an encoder comparison. `Sconv` scales S
# to hold s equal, and the unscaled arm is kept so the size of that effect is visible.
function run_conv(agg, credit, seed; Sconv=S)
    m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T, S=Sconv, L=L, LF=LF)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        conv_train!(m, Ptr, Ytr, rng, agg, credit)
        a = accuracy([conv_predict(m, p, agg) for p in Pte], Yte)
        a > best && (best = a)
    end
    return best
end

const S_MATCHED = max(1, round(Int, WIDTH / (784 / S)))   # same s as the flat baseline

arms = (("flat FPTM  [baseline]", nothing, nothing, S),
        ("conv, max, random patch", :max, :random, S),
        ("conv, max, argmax patch", :max, :argmax, S),
        ("conv, sum, argmax patch", :sum, :argmax, S),
        ("conv, max, random, s matched", :max, :random, S_MATCHED),
        ("conv, max, argmax, s matched", :max, :argmax, S_MATCHED))

println("arm                          accuracy (per seed)")
println("-"^92)
res = Dict{String,Vector{Float64}}()
for (name, agg, credit, Sarm) in arms
    accs = Float64[]
    for sd in SEEDS
        t = @elapsed (a = agg === nothing ? run_flat(sd) : run_conv(agg, credit, sd; Sconv=Sarm))
        push!(accs, a)
        @printf("%-28s seed %d  %.4f   (%.0f s)\n", name, sd, a, t)
    end
    res[name] = accs
end

println()
println("="^92)
base = mean(res["flat FPTM  [baseline]"])
for (name, _, _) in arms
    @printf("%-28s mean %.4f   %+.4f vs flat\n", name, mean(res[name]), mean(res[name]) - base)
end
println()
println("Gate: a conv arm must beat flat FPTM at the same clause count on identical booleanization.")
