# Graded messages: is a clause vote's MAGNITUDE worth more than more clauses?
#
# The design notes call this "the real research contribution" of the fuzzy/graph combination, and
# also note the cost: thermometer-encoding a vote into k levels multiplies message width by k. That
# cost is what makes the obvious experiment worthless. Comparing 24 binary message bits against 24
# graded ones from 24 clauses is comparing a 24-bit channel to a 96-bit one, and of course the wider
# channel wins.
#
# So every arm here gets exactly 24 message bits per edge, and spends them differently:
#
#     grades  source clauses  bits/clause     what one message bit means
#     1       24              1               "clause j fired"          (binary - Stage 5 baseline)
#     2       12              2               "clause j scored >= t"    for t in 1:2
#     4        6              4               "clause j scored >= t"    for t in 1:4
#
# Identical node width, identical classifier, identical parameters. The only thing that changes is
# whether the channel spends its bits on more clauses or on finer resolution per clause.
#
# WHAT A CLAUSE VOTE ACTUALLY GRADES. Not "how many symbols of set B are here" - the vote is the
# ceiling minus the number of included literals that missed, so a clause with 8 includes on a node
# holding 4 symbols never fires at all. What it grades is how completely one specific pattern is
# present. Any task meant to reward graded messages has to be built on that, not on counting.
#
# TWO TASKS, because one result here is unreadable. A win on a task built to reward magnitude says
# nothing; the claim only becomes falsifiable if there is also a task where graded should lose.
#
#   PREDICATE task - Stage 5's adjacency task. A symbol from set A followed by one from set B. The
#   useful message is one bit, "am I in B?". Graded resolution has nothing to encode, and it costs
#   three quarters of the source clauses, so it should lose or tie.
#
#   MAGNITUDE task - nodes hold 4 symbols each. Label = some node overlaps pattern P in >= 3 of 4
#   positions AND its right neighbour overlaps pattern Q in >= 3. Negatives contain both such nodes
#   but never adjacent in that order, so presence is matched and only the overlap degree across the
#   edge separates the classes. That is exactly what a clause vote measures.
#
# The fight is fairer than it looks: a binary clause can already pick its own threshold by sizing its
# include set, since it fires when overlap > n_included - LF. Binary gets four times as many clauses
# to spend on different thresholds; graded gets several thresholds out of one clause. Which is the
# better use of 24 bits is not obvious in either direction.
#
#   julia --project=. research/graded-messages/run.jl [epochs] [seeds]

using Printf, Random, Statistics
using TMCore

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
const NSEEDS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const SEEDS = (20260908, 11, 12, 13, 14)[1:NSEEDS]

const K = 32
const NLEN = 10
const NTRAIN, NTEST = 6_000, 3_000
const NCL, T, S, L, LF = 24, 8, 40, 10, 5
const MSGW = 24                 # message bits per edge, held constant across every arm
const MW = K + 2MSGW            # node width, likewise constant

# --- task 1: predicate. A symbol from SETA immediately followed by one from SETB. ---------------
const SETA, SETB = 1:4, 5:8

function seq_predicate(rng, positive::Bool)
    while true
        s = rand(rng, 9:K, NLEN)
        if positive
            i = rand(rng, 1:(NLEN - 1))
            s[i] = rand(rng, SETA); s[i + 1] = rand(rng, SETB)
        else
            i = rand(rng, 1:NLEN)
            js = [j for j in 1:NLEN if j != i && j != i + 1]
            isempty(js) && continue
            s[i] = rand(rng, SETA); s[rand(rng, js)] = rand(rng, SETB)
        end
        any(k -> s[k] in SETA && s[k+1] in SETB, 1:(NLEN-1)) == positive && return s
    end
end

# --- task 2: magnitude. Overlap degree with patterns P and Q across an edge. --------------------
const PATP, PATQ = 1:6, 7:12
const FILL = 13:K
const NSYM = 4                  # symbols per node

ov(node, pat) = count(x -> x in pat, node)

"A node holding `k` symbols drawn from `pat` and the rest from FILL."
function mknode(rng, pat, k)
    vcat(shuffle(rng, collect(pat))[1:k], shuffle(rng, collect(FILL))[1:(NSYM - k)])
end

function seq_magnitude(rng, positive::Bool)
    while true
        # Background nodes reach overlap 2, one short of the threshold. A 3-versus-1 gap made the
        # task solvable by any thresholding clause and left no headroom above the binary arm; the
        # question is only interesting where a single symbol separates the classes.
        s = [mknode(rng, rand(rng, Bool) ? PATP : PATQ, rand(rng, 0:2)) for _ in 1:NLEN]
        if positive
            i = rand(rng, 1:(NLEN - 1))
            s[i] = mknode(rng, PATP, 3)
            s[i + 1] = mknode(rng, PATQ, 3)
        else
            # both high-overlap nodes present, never adjacent in the P-then-Q order
            j = rand(rng, 1:NLEN)
            ks = [k for k in 1:NLEN if k != j + 1 && k != j]
            isempty(ks) && continue
            s[j] = mknode(rng, PATP, 3)
            s[rand(rng, ks)] = mknode(rng, PATQ, 3)
        end
        hit = any(k -> ov(s[k], PATP) >= 3 && ov(s[k+1], PATQ) >= 3, 1:(NLEN-1))
        hit == positive && return s
    end
end

function dataset(gen, rng, n)
    S = Vector{Any}(undef, n); Y = Vector{Bool}(undef, n)
    for i in 1:n
        y = rand(rng, Bool); S[i] = gen(rng, y); Y[i] = y
    end
    return S, Y
end

# Task 1 nodes are one symbol, task 2 nodes a set of four; multi-hot covers both.
tobits(node::Int) = [i == node for i in 1:K]
tobits(node::Vector{Int}) = [i in node for i in 1:K]

# --- per-node machinery: max over nodes, uniform credit (Stage 3's answer) ----------------------
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

# --- static arms: no messages, and the identity ceiling ----------------------------------------
nodes_plain(s) = [TMInput(tobits(x)) for x in s]

nodes_identity(s) =
    [TMInput(Vector{Bool}(vcat(tobits(s[i]),
                               i > 1 ? tobits(s[i-1]) : falses(K),
                               i < NLEN ? tobits(s[i+1]) : falses(K)))) for i in 1:NLEN]

function run_static(Str, Ytr, Ste, Yte, build, width, seed)
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

# --- graded message channel ---------------------------------------------------------------------
"""
Thermometer-encode each source clause's vote into `gr` bits: bit `t` is set iff the vote is at least
`t`. `gr == 1` reduces to "the clause fired", the binary baseline, so both live on one code path
rather than in two implementations that could drift apart.

`nmsg * gr` is held at MSGW, so every arm sends the same number of bits per edge.
"""
function build_graded(s, src, nmsg, gr, lfv, cpv)
    raw = [TMInput(Vector{Bool}(vcat(tobits(x), falses(2MSGW)))) for x in s]
    bits = falses(NLEN, MSGW)
    @inbounds for i in 1:NLEN, j in 1:nmsg
        v = clause_vote(src, j, raw[i], lfv, cpv)
        for t in 1:gr
            bits[i, (j - 1) * gr + t] = v >= t
        end
    end
    out = Vector{TMInput}(undef, NLEN)
    @inbounds for i in 1:NLEN
        v = falses(MW)
        v[1:K] .= tobits(s[i])
        i > 1 && (v[K+1:K+MSGW] .= bits[i-1, :])
        i < NLEN && (v[K+MSGW+1:MW] .= bits[i+1, :])
        out[i] = TMInput(Vector{Bool}(v))
    end
    return out
end

"""
The classifier's own positive bank is the message source, so round-1 feedback reshapes the channel
as a side effect - GraphTM's depth mechanism, one set of automata serving both depths.

Returns accuracy, the mean vote of a firing source clause, and the fraction of firing votes that are
strictly below the ceiling. That last number decides whether the arm was testable at all: if source
votes always sit at the ceiling the thermometer bits are perfectly correlated and the extra levels
are dead weight by construction rather than by result.
"""
function run_graded(Str, Ytr, Ste, Yte, gr, seed)
    nmsg = MSGW ÷ gr
    m = TMClassifier(Ytr, MW; clauses_per_class=2NCL, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(seed)
    lfv, cpv = m.params.LF, m.ceiling
    best = 0.0
    for _ in 1:EPOCHS
        src = m.positive[1]
        Ntr = [build_graded(s, src, nmsg, gr, lfv, cpv) for s in Str]
        for i in randperm(rng, length(Ytr))
            yi = findfirst(==(Ytr[i]), m.classes)
            node_update!(m, yi, Ntr[i], true, rng)
            for ci in eachindex(m.classes)
                ci == yi || node_update!(m, ci, Ntr[i], false, rng)
            end
        end
        Nte = [build_graded(s, m.positive[1], nmsg, gr, lfv, cpv) for s in Ste]
        a = accuracy([node_predict(m, n, rng) for n in Nte], Yte)
        a > best && (best = a)
    end
    src = m.positive[1]
    vs = Int[]
    for s in Ste[1:200]
        for x in build_graded(s, src, nmsg, gr, lfv, cpv), j in 1:nmsg
            v = clause_vote(src, j, x, lfv, cpv)
            v > 0 && push!(vs, v)
        end
    end
    # Include count of the source clauses. Under the LiteralCapped ceiling a clause's vote cannot
    # exceed min(n_included, LF), so a bank of one- and two-literal clauses puts levels above 2 out
    # of reach *by construction*. Without this number a flat result is unreadable: it could equally
    # mean graded resolution does not pay, or that this run never offered any to test.
    inc = mean(Int(src.count[j]) for j in 1:nmsg)
    return best, (isempty(vs) ? 0.0 : mean(vs)), inc
end

# --- run ------------------------------------------------------------------------------------
tasks = (("PREDICATE  (A then B adjacent)", seq_predicate),
         ("MAGNITUDE  (overlap >=3 with P then Q)", seq_magnitude))

const ARMS = ("no messages  [control]", "identity     [ceiling]", "binary   24 clauses x 1",
              "graded   12 clauses x 2", "graded    6 clauses x 4")

println("="^96)
println("Graded messages at equal channel width: magnitude per clause, or more clauses?")
println("="^96)
@printf("alphabet %d, length %d, %d train, %d test, %d clauses/polarity, LF %d\n",
        K, NLEN, NTRAIN, NTEST, NCL, LF)
@printf("message bits per edge held at %d in every message arm; node width %d\n", MSGW, MW)
@printf("%d epochs, %d seeds\n\n", EPOCHS, length(SEEDS))

function main()
    results = Dict{Tuple{String,String},Vector{Float64}}()
    for (tname, gen) in tasks
        rng0 = MersenneTwister(4242)
        Str, Ytr = dataset(gen, rng0, NTRAIN)
        Ste, Yte = dataset(gen, rng0, NTEST)

        # The classes must be matched on node content, or nothing below is about messages.
        bag(s) = [any(x -> (x isa Int ? x == k : k in x), s) for k in 1:K]
        pa = mean(hcat([bag(Str[i]) for i in eachindex(Str) if Ytr[i]]...), dims=2)
        pb = mean(hcat([bag(Str[i]) for i in eachindex(Str) if !Ytr[i]]...), dims=2)
        println("-"^96)
        @printf("%s   balance %.3f, max symbol-presence gap between classes %.4f\n",
                tname, mean(Ytr), maximum(abs.(pa .- pb)))
        println("-"^96)

        arms = ((ARMS[1], :static, nodes_plain, K, 0),
                (ARMS[2], :static, nodes_identity, 3K, 0),
                (ARMS[3], :graded, nothing, MW, 1),
                (ARMS[4], :graded, nothing, MW, 2),
                (ARMS[5], :graded, nothing, MW, 4))

        for (aname, kind, build, width, gr) in arms
            accs = Float64[]
            for sd in SEEDS
                if kind === :static
                    t = @elapsed (a = run_static(Str, Ytr, Ste, Yte, build, width, sd))
                    @printf("  %-24s seed %d  %.4f   (%.0f s)\n", aname, sd, a, t)
                else
                    t = @elapsed (r = run_graded(Str, Ytr, Ste, Yte, gr, sd))
                    a = r[1]
                    @printf("  %-24s seed %d  %.4f   source vote %.2f, ceiling min(%.2f, %d)  (%.0f s)\n",
                            aname, sd, a, r[2], r[3], LF, t)
                end
                push!(accs, a)
            end
            results[(tname, aname)] = accs
        end
        println()
    end

    println("="^96)
    println("SUMMARY")
    println("="^96)
    for (tname, _) in tasks
        println(tname)
        for aname in ARMS
            haskey(results, (tname, aname)) || continue
            @printf("  %-24s  mean %.4f\n", aname, mean(results[(tname, aname)]))
        end
        println()
    end
    println("Read the PREDICATE task first. Graded resolution has nothing to encode there, so if it")
    println("wins anyway the arms differ in something other than what this script claims to vary.")
end

main()
