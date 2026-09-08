# Does message passing earn anything on a real dataset?
#
# Every graph result in this repo so far rests on a synthetic task. `message-round` and
# `learned-messages` both built the task around the mechanism under test, which is the right way to
# ask whether a mechanism *can* work and says nothing about whether it does on data nobody designed.
# The design notes name the benchmark for this: CIFAR-10, where GraphTM reports +3.86 points over a
# convolutional TM.
#
# WHAT THIS CANNOT DO. Reproduce that number. Their convolutional baseline is in the 60s; flat FPTM
# on the booleanization available here reaches 0.33, so the absolute figures are not comparable to
# the paper and no comparison to it is drawn below. What is internally valid is the relative
# question, asked under one booleanization held fixed across every arm:
#
#     does one round of learned messages between patches beat convolution alone on real data?
#
# ARMS
#   flat FPTM            [baseline]  whole image, no patches
#   conv, no messages    [control]   patches, no edges - Stage 3's winner, and the arm to beat
#   conv + random msgs   [control]   a frozen random channel of the same width as the learned one
#   conv + learned msgs              the classifier's own bank as message source, as in Stage 5
#
# The random-message arm is the one that makes the table readable. Learned messages add 32 bits per
# patch, and without a random channel of identical width "messages help" cannot be told apart from
# "32 more bits help".
#
# EDGES. Patches sit on a grid; each is joined to its four neighbours, and the direction is the edge
# type, so a patch receives four separate message fields rather than one merged one. Direction has to
# be preserved or the messages cannot express anything spatial, which on images is the whole point.
#
#   julia --project=. research/cifar-messages/run.jl [epochs] [seeds]

using Printf, Random, Statistics
using TMCore, TMBoolean

include(joinpath(@__DIR__, "..", "cifar.jl"))

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
const NSEEDS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const SEEDS = (20260908, 11, 12, 13, 14)[1:NSEEDS]
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))

const NTRAIN = 20_000
const NTEST = 5_000
const DIM = 32
const NTH = 4                       # thermometer thresholds per pixel
const PATCH, STRIDE = 8, 6
const POS = 0:STRIDE:(DIM - PATCH)
const NPOS = length(POS)
const NPATCH = NPOS * NPOS
const PATCH_BITS = PATCH * PATCH * NTH
const POSBITS = 2 * (NPOS - 1)
const NMSG = 8                      # source clauses; 4 directions => 4*NMSG message bits
const MSGBITS = 4 * NMSG
const W_PLAIN = PATCH_BITS + POSBITS
const W_MSG = W_PLAIN + MSGBITS
const FLATW = DIM * DIM * NTH

const CLAUSES, T, S, L, LF = 40, 10, 125, 10, 5

println("="^96)
println("Message passing on CIFAR-10: does it beat convolution alone on real data?")
println("="^96)
@printf("%dx%d grayscale, %d thermometer thresholds -> %d flat bits\n", DIM, DIM, NTH, FLATW)
@printf("patch %d stride %d -> %d patches; width %d plain (%d pixel + %d position), %d with messages\n",
        PATCH, STRIDE, NPATCH, W_PLAIN, PATCH_BITS, POSBITS, W_MSG)
@printf("%d clauses/class, T %d, S %d, L %d, LF %d; %d train, %d test, %d epochs, %d seeds\n\n",
        CLAUSES, T, S, L, LF, NTRAIN, NTEST, EPOCHS, length(SEEDS))

# --- data --------------------------------------------------------------------------------------
print("loading CIFAR-10 ... ")
Xtr_raw, ytr_all = cifar10(DATA, :train)
Xte_raw, yte_all = cifar10(DATA, :test)
Gtr = cifar_gray(Xtr_raw[1:NTRAIN, :])
Gte = cifar_gray(Xte_raw[1:NTEST, :])
th = Thermometer(nthresholds=NTH, strategy=:quantile)
fit!(th, Gtr)
Btr, Bte = transform(th, Gtr), transform(th, Gte)
Ytr, Yte = Int.(ytr_all[1:NTRAIN]), Int.(yte_all[1:NTEST])
println("done")

# Thermometer columns are feature-major: pixel p occupies (p-1)*NTH+1 : p*NTH.
@inline pixbits(p) = ((p - 1) * NTH + 1):(p * NTH)

"""
Patch pixel bits plus a thermometer position code, written into `v` starting at offset 0.

Thermometer rather than one-hot for the position: the bits nest, so a clause can say "row >= 2" with
one literal instead of a disjunction it cannot express.
"""
function fill_patch!(v, row, yi, xi)
    y, x = POS[yi], POS[xi]
    k = 0
    @inbounds for r in 0:(PATCH - 1), c in 0:(PATCH - 1)
        p = (y + r) * DIM + (x + c) + 1
        for b in pixbits(p)
            k += 1
            v[k] = row[b]
        end
    end
    @inbounds for i in 1:(NPOS - 1)
        v[PATCH_BITS + i] = yi > i
        v[PATCH_BITS + (NPOS - 1) + i] = xi > i
    end
    return v
end

function patches_plain(row)
    out = Vector{TMInput}(undef, NPATCH)
    k = 0
    for yi in 1:NPOS, xi in 1:NPOS
        k += 1
        out[k] = TMInput(fill_patch!(falses(W_PLAIN), row, yi, xi))
    end
    return out
end

"Raw patches at message width, message field zeroed — the round-0 input."
function patches_raw_msgwidth(row)
    out = Vector{TMInput}(undef, NPATCH)
    k = 0
    for yi in 1:NPOS, xi in 1:NPOS
        k += 1
        out[k] = TMInput(fill_patch!(falses(W_MSG), row, yi, xi))
    end
    return out
end

const FLAT_TR = [TMInput(Vector{Bool}(view(Btr, i, :))) for i in 1:NTRAIN]
const FLAT_TE = [TMInput(Vector{Bool}(view(Bte, i, :))) for i in 1:NTEST]

# --- per-patch machinery: max aggregation, uniform credit (Stage 3's answers) --------------------
@inline function clause_over_patches(b, j, patches, LF, cp, rng)
    best, chosen, nfired = 0, 0, 0
    @inbounds for k in eachindex(patches)
        v = clause_vote(b, j, patches[k], LF, cp)
        v == 0 && continue
        nfired += 1
        rand(rng) * nfired < 1 && (chosen = k)
        v > best && (best = v)
    end
    return (best, chosen)
end

function conv_score(m, ci, patches, rng)
    LF, cp = m.params.LF, m.ceiling
    pos = neg = 0
    @inbounds for j in 1:m.positive[ci].nclauses
        pos += clause_over_patches(m.positive[ci], j, patches, LF, cp, rng)[1]
    end
    @inbounds for j in 1:m.negative[ci].nclauses
        neg += clause_over_patches(m.negative[ci], j, patches, LF, cp, rng)[1]
    end
    return pos - neg
end

function conv_predict(m, patches, rng)
    bi, bv = 1, typemin(Int)
    for ci in eachindex(m.classes)
        v = conv_score(m, ci, patches, rng)
        v > bv && ((bv, bi) = (v, ci))
    end
    return m.classes[bi]
end

function conv_update!(m, ci, patches, positive, rng)
    p = m.params
    Tt, LFv, Lv, sv, cp = p.T, p.LF, p.L, p.s, m.ceiling
    v = clamp(conv_score(m, ci, patches, rng), -Tt, Tt)
    upd = (positive ? (Tt - v) : (Tt + v)) / (2Tt)
    tI  = positive ? m.positive[ci] : m.negative[ci]
    tII = positive ? m.negative[ci] : m.positive[ci]
    @inbounds for j in 1:tI.nclauses
        rand(rng) < upd || continue
        _, k = clause_over_patches(tI, j, patches, LFv, cp, rng)
        n = Int(tI.count[j])
        if k != 0
            feedback!(TypeIa(), tI, j, patches[k], TMCore.reinforce_allowed(m.budget, n, Lv),
                      TMCore.promotion_room(m.budget, n, Lv, TypeIa()))
        else
            feedback!(TypeIb(), tI, j, sv, rng)
        end
    end
    @inbounds for j in 1:tII.nclauses
        rand(rng) < upd || continue
        _, k = clause_over_patches(tII, j, patches, LFv, cp, rng)
        k == 0 && continue
        feedback!(TypeII(), tII, j, patches[k],
                  TMCore.promotion_room(m.budget, Int(tII.count[j]), Lv, TypeII()))
    end
end

function epoch!(m, P, Y, rng)
    for i in randperm(rng, length(Y))
        yi = findfirst(==(Y[i]), m.classes)
        conv_update!(m, yi, P[i], true, rng)
        for ci in eachindex(m.classes)
            ci == yi || conv_update!(m, ci, P[i], false, rng)
        end
    end
end

# --- arms ---------------------------------------------------------------------------------------
function run_flat(seed)
    m = TMClassifier(Ytr, FLATW; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        train!(m, FLAT_TR, Ytr; rng=rng)
        a = accuracy(predict(m, FLAT_TE), Yte)
        a > best && (best = a)
    end
    return best, 0.0
end

function run_conv(seed)
    Ptr = [patches_plain(view(Btr, i, :)) for i in 1:NTRAIN]
    Pte = [patches_plain(view(Bte, i, :)) for i in 1:NTEST]
    m = TMClassifier(Ytr, W_PLAIN; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        epoch!(m, Ptr, Ytr, rng)
        a = accuracy([conv_predict(m, p, rng) for p in Pte], Yte)
        a > best && (best = a)
    end
    return best, 0.0
end

"Grid index of patch `k`'s neighbour in direction `d`, or 0 at the border."
@inline function neighbour(k, d)
    yi, xi = fldmod1(k, NPOS)
    d == 1 && (yi -= 1); d == 2 && (yi += 1); d == 3 && (xi -= 1); d == 4 && (xi += 1)
    (yi < 1 || yi > NPOS || xi < 1 || xi > NPOS) && return 0
    return (yi - 1) * NPOS + xi
end

"""
Attach one round of messages. A patch receives its four neighbours' round-0 clause outputs, one
`NMSG`-bit field per direction; a border patch receives zeros there, which is distinguishable from
any real message because a patch that fires nothing is rare.

Message inputs are zero at round 0, so a source clause's round-0 output depends on pixel and position
bits only. That is what lets one bank serve both depths, as GraphTM does.
"""
function build_messaged(row, src, lfv, cpv)
    raw = patches_raw_msgwidth(row)
    fired = falses(NPATCH, NMSG)
    @inbounds for k in 1:NPATCH, j in 1:NMSG
        fired[k, j] = clause_vote(src, j, raw[k], lfv, cpv) > 0
    end
    out = Vector{TMInput}(undef, NPATCH)
    @inbounds for k in 1:NPATCH
        v = falses(W_MSG)
        yi, xi = fldmod1(k, NPOS)
        fill_patch!(v, row, yi, xi)
        for d in 1:4
            nb = neighbour(k, d)
            nb == 0 && continue
            off = W_PLAIN + (d - 1) * NMSG
            for j in 1:NMSG
                v[off + j] = fired[nb, j]
            end
        end
        out[k] = TMInput(v)
    end
    return out
end

"""
`learned = false` freezes a randomly initialised bank as the channel. That control is what separates
"messages help" from "32 more bits help", and without it the learned arm is uninterpretable.
"""
function run_messaged(learned::Bool, seed)
    m = TMClassifier(Ytr, W_MSG; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(seed)
    lfv, cpv = m.params.LF, m.ceiling
    frozen = nothing
    if !learned
        frozen = TMClassifier(Ytr, W_MSG; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
        fb = frozen.positive[1]
        for j in 1:NMSG, _ in 1:8
            fb.state[rand(rng, 1:PATCH_BITS), j] = fb.include_limit
        end
        for j in 1:fb.nclauses
            for n in 1:fb.nchunks
                msk = zero(UInt64)
                for bit in 0:min(63, fb.width - (n - 1) * 64 - 1)
                    fb.state[(n - 1) * 64 + bit + 1, j] >= fb.include_limit && (msk |= one(UInt64) << bit)
                end
                fb.included[n, j] = msk
            end
            TMCore.recount!(fb, j)
        end
    end
    best = 0.0
    for _ in 1:EPOCHS
        src = learned ? m.positive[1] : frozen.positive[1]
        Ptr = [build_messaged(view(Btr, i, :), src, lfv, cpv) for i in 1:NTRAIN]
        epoch!(m, Ptr, Ytr, rng)
        src = learned ? m.positive[1] : frozen.positive[1]
        Pte = [build_messaged(view(Bte, i, :), src, lfv, cpv) for i in 1:NTEST]
        a = accuracy([conv_predict(m, p, rng) for p in Pte], Yte)
        a > best && (best = a)
    end
    # Fraction of source-clause outputs that are set. A silent channel and a useless one produce the
    # same accuracy, and only this number tells them apart.
    src = learned ? m.positive[1] : frozen.positive[1]
    fires = Bool[]
    for i in 1:200, x in patches_raw_msgwidth(view(Bte, i, :)), j in 1:NMSG
        push!(fires, clause_vote(src, j, x, lfv, cpv) > 0)
    end
    return best, mean(fires)
end

"""
Convolution at message width with the message field wired permanently to zero.

This is the control the learned arm lives or dies by, and it exists because the first run measured
the learned channel at 0.001 live — a field that is almost always zero is not sending anything, but
it is not inert either. Every message literal is then satisfiable by its negation for free, which
inflates a clause's literal count, which moves the `L` growth gate, which `budget-paths` already
showed is worth 13 points when it moves. Without this arm, "messages helped" cannot be separated from
"32 dead bits perturbed the literal budget".
"""
function run_dead(seed)
    build(row) = begin
        out = Vector{TMInput}(undef, NPATCH)
        k = 0
        for yi in 1:NPOS, xi in 1:NPOS
            k += 1
            out[k] = TMInput(fill_patch!(falses(W_MSG), row, yi, xi))
        end
        out
    end
    Ptr = [build(view(Btr, i, :)) for i in 1:NTRAIN]
    Pte = [build(view(Bte, i, :)) for i in 1:NTEST]
    m = TMClassifier(Ytr, W_MSG; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        epoch!(m, Ptr, Ytr, rng)
        a = accuracy([conv_predict(m, p, rng) for p in Pte], Yte)
        a > best && (best = a)
    end
    return best, 0.0
end

arms = (("flat FPTM      [baseline]", run_flat, FLATW),
        ("conv, no messages [control]", run_conv, W_PLAIN),
        ("conv + DEAD channel [control]", run_dead, W_MSG),
        ("conv + random msgs [control]", s -> run_messaged(false, s), W_MSG),
        ("conv + LEARNED msgs", s -> run_messaged(true, s), W_MSG))

println("arm                             accuracy per seed")
println("-"^96)
function main()
    res = Dict{String,Vector{Float64}}()
    for (name, f, width) in arms
        accs = Float64[]
        for sd in SEEDS
            t = @elapsed (r = f(sd))
            push!(accs, r[1])
            if r[2] > 0
                @printf("  %-30s seed %d  %.4f   msg bits live %.3f  (%.0f s)\n",
                        name, sd, r[1], r[2], t)
            else
                @printf("  %-30s seed %d  %.4f   (%.0f s)\n", name, sd, r[1], t)
            end
        end
        res[name] = accs
    end

    println()
    println("="^96)
    base = mean(res["conv, no messages [control]"])
    for (name, _, width) in arms
        @printf("%-30s mean %.4f   %+.4f vs conv   (width %d)\n",
                name, mean(res[name]), mean(res[name]) - base, width)
    end
    println()
    println("The arm to beat is conv-without-messages, not flat. Beating flat only says patches help,")
    println("which Stage 3 already established; beating the random-message arm is what says the")
    println("learned channel carries something rather than merely widening the input.")
end

main()
