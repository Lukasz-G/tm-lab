# Stage 4: one message round, thresholded Boolean messages.
#
# The benchmark trap in the design notes: on Amazon Sales, flat FPTM beats GraphTM at every noise
# level, which makes that comparison uninformative about whether topology contributes anything. So
# the task here is constructed so that a flat model provably cannot win, and the controls are the
# point.
#
# TASK. Sequences of `K` symbols. The label is whether the bigram (1,2) appears **adjacently**.
# Negatives contain both symbol 1 and symbol 2 but never next to each other. So the two classes have
# identical symbol *presence*, and a bag-of-symbols model is at chance by construction rather than by
# luck. Only adjacency separates them.
#
# ARMS, in increasing access to structure:
#   flat, bag of symbols      presence of each symbol anywhere      -> must be chance
#   flat, positional          the whole sequence one-hot per slot   -> can learn it, position by
#                                                                      position, which is the
#                                                                      inefficiency graphs claim to fix
#   per-node, no messages     each node sees its own symbol         -> must be chance: a node alone
#                                                                      cannot see a pair
#   per-node, one message     node sees its own symbol plus its
#                             left and right neighbours' outputs    -> should solve it
#
# The first and third arms are controls. If either scores above chance the task is not what it claims
# and nothing else in the table can be read.
#
# Messages here are the *identity* message — a neighbour's symbol, thresholded — which is what one
# round with an untrained message clause reduces to. Learned message clauses are Stage 5 territory.
#
#   julia --project=. research/message-round/run.jl [epochs] [seeds]

using Printf, Random, Statistics
using TMCore

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
const NSEEDS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const SEEDS = (20260908, 11, 12, 13, 14)[1:NSEEDS]

const K = 8            # alphabet
const NLEN = 12        # sequence length
const NTRAIN, NTEST = 10_000, 5_000
const CLAUSES, T, S, L, LF = 40, 10, 40, 10, 5

"""
Generate one sequence and its label.

Positive: symbols 1 and 2 adjacent somewhere. Negative: both present, never adjacent. Constructed
rather than filtered, so the classes match on symbol presence exactly and the bag-of-symbols control
is at chance by design.
"""
function make_sequence(rng, positive::Bool)
    while true
        s = rand(rng, 3:K, NLEN)                 # fill with symbols that are neither 1 nor 2
        i = rand(rng, 1:(NLEN - 1))
        if positive
            s[i] = 1; s[i + 1] = 2
        else
            # place 1 and 2 at least two apart
            j = rand(rng, 1:NLEN)
            ks = [k for k in 1:NLEN if abs(k - j) > 1]
            isempty(ks) && continue
            s[j] = 1; s[rand(rng, ks)] = 2
        end
        # Reject positives that also fail, and negatives that accidentally became positive.
        adj = any(k -> (s[k] == 1 && s[k+1] == 2) || (s[k] == 2 && s[k+1] == 1), 1:(NLEN-1))
        adj == positive && return s
    end
end

function dataset(rng, n)
    S = Vector{Vector{Int}}(undef, n)
    Y = Vector{Bool}(undef, n)
    for i in 1:n
        y = rand(rng, Bool)
        S[i] = make_sequence(rng, y)
        Y[i] = y
    end
    return S, Y
end

rng0 = MersenneTwister(1234)
Str, Ytr = dataset(rng0, NTRAIN)
Ste, Yte = dataset(rng0, NTEST)

println("="^92)
println("Stage 4 — one message round, on a task where messages are load-bearing")
println("="^92)
@printf("alphabet %d, length %d, %d train, %d test, %d clauses/class, %d epochs, %d seeds\n",
        K, NLEN, NTRAIN, NTEST, CLAUSES, EPOCHS, length(SEEDS))
@printf("class balance: %.3f positive\n", count(Ytr) / length(Ytr))

# Sanity: the two classes must be indistinguishable by symbol presence.
pres(s) = [any(==(k), s) for k in 1:K]
pin = mean(hcat([pres(Str[i]) for i in eachindex(Str) if Ytr[i]]...), dims=2)
pout = mean(hcat([pres(Str[i]) for i in eachindex(Str) if !Ytr[i]]...), dims=2)
@printf("max difference in symbol presence between classes: %.4f  (0 = perfectly matched)\n\n",
        maximum(abs.(pin .- pout)))

onehot(sym) = [i == sym for i in 1:K]

# --- feature builders, one per arm -------------------------------------------------------------
flat_bag(s) = TMInput(pres(s))
flat_pos(s) = TMInput(vcat([onehot(x) for x in s]...))

"Nodes with their own symbol only."
nodes_plain(s) = [TMInput(onehot(x)) for x in s]

"""
Nodes carrying their own symbol plus one round of messages from the left and right neighbours.

A missing neighbour at either end sends an all-zero message, which is the honest encoding of "no
edge" — it is distinguishable from every real symbol because one-hot codes always have a bit set.
"""
function nodes_msg(s)
    out = Vector{TMInput}(undef, NLEN)
    for i in 1:NLEN
        v = falses(3K)
        v[1:K] .= onehot(s[i])
        i > 1 && (v[K+1:2K] .= onehot(s[i-1]))
        i < NLEN && (v[2K+1:3K] .= onehot(s[i+1]))
        out[i] = TMInput(Vector{Bool}(v))
    end
    return out
end

# --- per-node model, max over nodes (Stage 3 found max beats sum) ------------------------------
@inline function clause_over_nodes(b, j, nodes, LF, cp, rng)
    best, chosen, nfired = 0, 0, 0
    @inbounds for k in eachindex(nodes)
        v = clause_vote(b, j, nodes[k], LF, cp)
        v == 0 && continue
        nfired += 1
        rand(rng) * nfired < 1 && (chosen = k)     # uniform over firing nodes, as the reference does
        v > best && (best = v)
    end
    return (best, chosen)
end

function node_score(m, ci, nodes, rng)
    LF, cp = m.params.LF, m.ceiling
    pos = neg = 0
    @inbounds for j in 1:m.positive[ci].nclauses
        pos += clause_over_nodes(m.positive[ci], j, nodes, LF, cp, rng)[1]
    end
    @inbounds for j in 1:m.negative[ci].nclauses
        neg += clause_over_nodes(m.negative[ci], j, nodes, LF, cp, rng)[1]
    end
    return pos - neg
end

function node_predict(m, nodes, rng)
    bi, bv = 1, typemin(Int)
    for ci in eachindex(m.classes)
        v = node_score(m, ci, nodes, rng)
        v > bv && ((bv, bi) = (v, ci))
    end
    return m.classes[bi]
end

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

function run_flat(build, width, seed)
    Xtr = [build(s) for s in Str]
    Xte = [build(s) for s in Ste]
    m = TMClassifier(Ytr, width; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(seed)
    best = 0.0
    for _ in 1:EPOCHS
        train!(m, Xtr, Ytr; rng=rng)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
    end
    return best
end

function run_nodes(build, width, seed)
    Ntr = [build(s) for s in Str]
    Nte = [build(s) for s in Ste]
    m = TMClassifier(Ytr, width; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
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

arms = (("flat, bag of symbols  [control]", :flat, flat_bag, K),
        ("flat, positional",                :flat, flat_pos, NLEN * K),
        ("per-node, no messages [control]", :node, nodes_plain, K),
        ("per-node, ONE message round",     :node, nodes_msg, 3K))

println("arm                                accuracy (per seed)")
println("-"^92)
res = Dict{String,Vector{Float64}}()
for (name, kind, build, width) in arms
    accs = Float64[]
    for sd in SEEDS
        t = @elapsed (a = kind === :flat ? run_flat(build, width, sd) : run_nodes(build, width, sd))
        push!(accs, a)
        @printf("%-34s seed %d  %.4f   (%.0f s)\n", name, sd, a, t)
    end
    res[name] = accs
end

println()
println("="^92)
for (name, _, _, width) in arms
    @printf("%-34s mean %.4f   (width %d)\n", name, mean(res[name]), width)
end
println()
println("Controls must sit at ~0.50. If either is above chance the task is not what it claims and")
println("nothing else here can be read.")
