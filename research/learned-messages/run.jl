# Do LEARNED messages beat a fixed neighbourhood window?
#
# Stage 4 showed one round of *identity* messages — a node receives its neighbours' features —
# solving an adjacency task perfectly. But identity messages are a 3-gram window with a graph label
# on it. What distinguishes GraphTM is that the symbols travelling along edges are **clause outputs**,
# learned jointly with the classifier.
#
# Stage 4's task cannot test this: identity messages already score 1.0000, so there is no headroom.
# The question only has teeth when the message channel is **too narrow to carry the raw features**,
# forcing a choice about what to send.
#
# TASK. Alphabet of 32 symbols, length 12. Label = some position holds a symbol from set A and the
# next position holds a symbol from set B (|A| = |B| = 4). Negatives contain symbols from both sets
# but never in that adjacency. So:
#   - the useful message from a right neighbour is one bit, "am I in B?"
#   - carrying it as identity costs 32 bits per edge
#   - a learned message clause could discover it and spend 1
#
# ARMS
#   no messages          [control]  must be chance: a node alone cannot see a pair
#   identity messages    [ceiling]  full 32-bit neighbour features, 96 bits/node
#   random messages      [control]  fixed random clauses as the channel — does ANY projection work?
#   learned messages                GraphTM's mechanism: one clause bank of width K + 2*nclauses,
#                                   evaluated twice. At round 0 the message inputs are zero, so a
#                                   clause's round-0 output is how it responds to raw features; those
#                                   outputs become the messages for round 1. The bank is trained by
#                                   round-1 feedback, and its round-0 behaviour shifts as a side
#                                   effect — which is exactly how GraphTM trains depth, since the
#                                   shallow and deep clauses are the same automata.
#
# The random arm is the one that matters. If learned only matches random, the channel is doing the
# work and "learned" is decoration.
#
#   julia --project=. research/learned-messages/run.jl [epochs] [seeds]

using Printf, Random, Statistics
using TMCore

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 25
const NSEEDS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const SEEDS = (20260908, 11, 12, 13, 14)[1:NSEEDS]

const K = 32
const NLEN = 12
const SETA, SETB = 1:4, 5:8
const NTRAIN, NTEST = 8_000, 4_000
const NCL, T, S, L, LF = 20, 8, 40, 10, 5     # NCL clauses per polarity; also the message width

function make_sequence(rng, positive::Bool)
    while true
        s = rand(rng, 9:K, NLEN)                        # neither A nor B
        if positive
            i = rand(rng, 1:(NLEN - 1))
            s[i] = rand(rng, SETA); s[i + 1] = rand(rng, SETB)
        else
            i = rand(rng, 1:NLEN)
            js = [j for j in 1:NLEN if j != i + 1 && j != i]
            isempty(js) && continue
            s[i] = rand(rng, SETA); s[rand(rng, js)] = rand(rng, SETB)
        end
        hit = any(k -> s[k] in SETA && s[k+1] in SETB, 1:(NLEN-1))
        hit == positive && return s
    end
end

function dataset(rng, n)
    S = Vector{Vector{Int}}(undef, n); Y = Vector{Bool}(undef, n)
    for i in 1:n
        y = rand(rng, Bool); S[i] = make_sequence(rng, y); Y[i] = y
    end
    return S, Y
end

rng0 = MersenneTwister(4242)
Str, Ytr = dataset(rng0, NTRAIN)
Ste, Yte = dataset(rng0, NTEST)

println("="^94)
println("Do learned messages beat a fixed neighbourhood window?")
println("="^94)
@printf("alphabet %d, length %d, A=%s B=%s, %d train, %d test, %d clauses, %d epochs, %d seeds\n",
        K, NLEN, string(SETA), string(SETB), NTRAIN, NTEST, NCL, EPOCHS, length(SEEDS))
@printf("class balance %.3f positive\n", mean(Ytr))
pres(s) = [any(==(k), s) for k in 1:K]
pa = mean(hcat([pres(Str[i]) for i in eachindex(Str) if Ytr[i]]...), dims=2)
pb = mean(hcat([pres(Str[i]) for i in eachindex(Str) if !Ytr[i]]...), dims=2)
@printf("max symbol-presence difference between classes: %.4f\n\n", maximum(abs.(pa .- pb)))

onehot(x) = [i == x for i in 1:K]

# --- node feature builders -----------------------------------------------------------------
nodes_plain(s) = [TMInput(onehot(x)) for x in s]

function nodes_identity(s)
    [TMInput(Vector{Bool}(vcat(onehot(s[i]),
                               i > 1 ? onehot(s[i-1]) : falses(K),
                               i < NLEN ? onehot(s[i+1]) : falses(K)))) for i in 1:NLEN]
end

# --- per-node machinery, following Stage 3: max over nodes, uniform credit ------------------
@inline function clause_over_nodes(b, j, nodes, LF, cp, rng)
    best, chosen, nfired = 0, 0, 0
    @inbounds for k in eachindex(nodes)
        v = clause_vote(b, j, nodes[k], LF, cp)
        v == 0 && continue
        nfired += 1
        rand(rng) * nfired < 1 && (chosen = k)
        v > best && (best = v)
    end
    return (best, chosen)
end

function node_score(m, ci, nodes, rng)
    LF, cp = m.params.LF, m.ceiling
    p = n = 0
    @inbounds for j in 1:m.positive[ci].nclauses
        p += clause_over_nodes(m.positive[ci], j, nodes, LF, cp, rng)[1]
    end
    @inbounds for j in 1:m.negative[ci].nclauses
        n += clause_over_nodes(m.negative[ci], j, nodes, LF, cp, rng)[1]
    end
    return p - n
end

node_predict(m, nodes, rng) =
    m.classes[argmax([node_score(m, ci, nodes, rng) for ci in eachindex(m.classes)])]

function node_update!(m, ci, nodes, positive, rng)
    p = m.params
    Tt, LFv, Lv, sv, cp = p.T, p.LF, p.L, p.s, m.ceiling
    v = clamp(node_score(m, ci, nodes, rng), -Tt, Tt)
    upd = (positive ? (Tt - v) : (Tt + v)) / (2Tt)
    tI  = positive ? m.positive[ci] : m.negative[ci]
    tII = positive ? m.negative[ci] : m.positive[ci]
    @inbounds for j in 1:tI.nclauses
        rand(rng) < upd || continue
        _, k = clause_over_nodes(tI, j, nodes, LFv, cp, rng)
        n = Int(tI.count[j])
        if k != 0
            feedback!(TypeIa(), tI, j, nodes[k], TMCore.reinforce_allowed(m.budget, n, Lv),
                      TMCore.promotion_room(m.budget, n, Lv, TypeIa()))
        else
            feedback!(TypeIb(), tI, j, sv, rng)
        end
    end
    @inbounds for j in 1:tII.nclauses
        rand(rng) < upd || continue
        _, k = clause_over_nodes(tII, j, nodes, LFv, cp, rng)
        k == 0 && continue
        feedback!(TypeII(), tII, j, nodes[k],
                  TMCore.promotion_room(m.budget, Int(tII.count[j]), Lv, TypeII()))
    end
end

function run_static(build, width, seed)
    Ntr = [build(s) for s in Str]; Nte = [build(s) for s in Ste]
    m = TMClassifier(Ytr, width; clauses_per_class=2NCL, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        for i in randperm(rng, length(Ytr))
            yi = findfirst(==(Ytr[i]), m.classes)
            node_update!(m, yi, Ntr[i], true, rng)
            for ci in eachindex(m.classes)
                ci == yi || node_update!(m, ci, Ntr[i], false, rng)
            end
        end
        a = accuracy([node_predict(m, n, rng) for n in Nte], Yte)
        a > best && (best = a)
    end
    return best
end

# --- messages produced by a clause bank ------------------------------------------------------
const MW = K + 2NCL           # node width once messages are attached

"""
Round-0 outputs of `msgbank` on each node, then round-1 node features: own symbol, left neighbour's
message bits, right neighbour's message bits. Message inputs are zero at round 0, so a clause's
round-0 response depends on raw features only — which is what makes the same bank usable at both
depths, as GraphTM does.
"""
function build_messaged(s, msgbank, LF, cp)
    raw = [TMInput(Vector{Bool}(vcat(onehot(x), falses(2NCL)))) for x in s]
    fired = falses(NLEN, NCL)
    @inbounds for i in 1:NLEN, j in 1:NCL
        fired[i, j] = clause_vote(msgbank, j, raw[i], LF, cp) > 0
    end
    out = Vector{TMInput}(undef, NLEN)
    @inbounds for i in 1:NLEN
        v = falses(MW)
        v[1:K] .= onehot(s[i])
        i > 1 && (v[K+1:K+NCL] .= fired[i-1, :])
        i < NLEN && (v[K+NCL+1:MW] .= fired[i+1, :])
        out[i] = TMInput(Vector{Bool}(v))
    end
    return out
end

"""
`learned = true` uses the classifier's own positive bank as the message source, so round-1 feedback
reshapes the messages as a side effect — GraphTM's depth mechanism. `false` freezes a randomly
initialised bank, which is the control that says whether learning the channel matters at all.
"""
function run_messaged(learned::Bool, seed)
    m = TMClassifier(Ytr, MW; clauses_per_class=2NCL, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(seed)
    lfv, cpv = m.params.LF, m.ceiling
    frozen = nothing
    if !learned
        # A random but fixed channel: include each literal with probability ~1/8.
        frozen = TMClassifier(Ytr, MW; clauses_per_class=2NCL, T=T, S=S, L=L, LF=LF)
        fb = frozen.positive[1]
        for j in 1:fb.nclauses, _ in 1:6
            i = rand(rng, 1:K)
            fb.state[i, j] = fb.include_limit
        end
        for j in 1:fb.nclauses
            TMCore.recount!(fb, j)
        end
        for j in 1:fb.nclauses
            for n in 1:fb.nchunks
                msk = zero(UInt64)
                for bit in 0:min(63, fb.width - (n-1)*64 - 1)
                    fb.state[(n-1)*64 + bit + 1, j] >= fb.include_limit && (msk |= one(UInt64) << bit)
                end
                fb.included[n, j] = msk
            end
            TMCore.recount!(fb, j)
        end
    end
    best = 0.0
    for _ in 1:EPOCHS
        src = learned ? m.positive[1] : frozen.positive[1]
        Ntr = [build_messaged(s, src, lfv, cpv) for s in Str]
        for i in randperm(rng, length(Ytr))
            yi = findfirst(==(Ytr[i]), m.classes)
            node_update!(m, yi, Ntr[i], true, rng)
            for ci in eachindex(m.classes)
                ci == yi || node_update!(m, ci, Ntr[i], false, rng)
            end
        end
        src = learned ? m.positive[1] : frozen.positive[1]
        Nte = [build_messaged(s, src, lfv, cpv) for s in Ste]
        a = accuracy([node_predict(m, n, rng) for n in Nte], Yte)
        a > best && (best = a)
    end
    # Diagnostic. If the classifier comes to depend on message literals, those literals can never be
    # satisfied at round 0 — where message inputs are zero by construction — so the source bank's
    # round-0 output collapses and the channel goes silent. Measuring both sides says whether that
    # is what happened rather than leaving it a guess.
    src = learned ? m.positive[1] : frozen.positive[1]
    live = mean([mean(hcat([[clause_vote(src, j, x, lfv, cpv) > 0 for j in 1:src.nclauses]
                            for x in build_messaged(s, src, lfv, cpv)]...)) for s in Ste[1:200]])
    msglits = 0
    for j in 1:m.positive[1].nclauses, i in (K+1):MW
        b = m.positive[1]
        n, bit = (i - 1) >> 6 + 1, (i - 1) & 63
        (!iszero(b.included[n, j] & (one(UInt64) << bit)) ||
         !iszero(b.included_inv[n, j] & (one(UInt64) << bit))) && (msglits += 1)
    end
    return (best, live, msglits)
end

arms = (("no messages        [control]", () -> nothing, :static, nodes_plain, K),
        ("identity messages  [ceiling]", () -> nothing, :static, nodes_identity, 3K),
        ("random messages    [control]", () -> nothing, :random, nothing, MW),
        ("LEARNED messages",             () -> nothing, :learned, nothing, MW))

println("arm                                accuracy (per seed)")
println("-"^94)
res = Dict{String,Vector{Float64}}()
for (name, _, kind, build, width) in arms
    accs = Float64[]
    for sd in SEEDS
        if kind === :static
            t = @elapsed (a = run_static(build, width, sd))
            push!(accs, a)
            @printf("%-34s seed %d  %.4f   (%.0f s)\n", name, sd, a, t)
        else
            t = @elapsed (r = run_messaged(kind === :learned, sd))
            push!(accs, r[1])
            @printf("%-34s seed %d  %.4f   msg bits live %.3f, msg literals used %d  (%.0f s)\n",
                    name, sd, r[1], r[2], r[3], t)
        end
    end
    res[name] = accs
end

println()
println("="^94)
for (name, _, _, _, width) in arms
    @printf("%-34s mean %.4f   (node width %d)\n", name, mean(res[name]), width)
end
println()
println("LEARNED must beat RANDOM, or the channel is doing the work and learning is decoration.")
