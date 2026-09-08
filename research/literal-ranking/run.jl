# Can a confidence gradient make an unreadable clause readable?
#
# imdb-readability ended in a dead end. The published one-clause-per-class IMDb model reaches 0.90
# with clauses of ~3,800 literals, and the extracted core is 3,780 of them: precise, and useless to
# a human. Worse, there was no way even to rank them — every included automaton sat on the include
# threshold, so all 3,780 literals looked equally important.
#
# eviction/ changed that. Reset eviction produces a real confidence gradient for 0 to 0.4 accuracy
# points. Weighting the *vote* by it does not pay (confidence-payoff/), but ranking does not need it
# to: it needs only that the gradient exist.
#
# So: train IMDb with reset eviction, rank each clause's literals by automaton state, and ask whether
# a top-N truncation keeps the full rule's precision. If twenty words do most of the work of 3,780,
# the dead end has an exit.
#
# Baseline to beat: ranking by *satisfaction frequency*, which needs no gradient at all and is what
# anyone would try first. Confidence has to beat that, not just beat random, or it adds nothing.
#
# Run research/imdb-readability/prepare.py first.
#
#   julia --project=. research/literal-ranking/run.jl [epochs]

using Printf, Random, Statistics
using TMCore
import TMCore: reinforce_allowed, promotion_room

const DATA = normpath(joinpath(@__DIR__, "..", "..", "data", "imdb"))
const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
const T, S, L, LF = 18, 1000, 64, 64
const CLAUSES = 2

struct PartialFreeze <: TMCore.LiteralBudgetPolicy end
reinforce_allowed(::PartialFreeze, ::Integer, ::Integer) = true
promotion_room(::PartialFreeze, n::Integer, L::Integer, ::TypeIa) = n <= L ? typemax(Int) : 0
promotion_room(::PartialFreeze, ::Integer, ::Integer, ::TypeII) = typemax(Int)

function erode!(b, j, s, rng, reset::Bool)
    il, smin = b.include_limit, b.state_min
    changed = false
    @inbounds for _ in 1:s
        for (arr, mask) in ((b.state, b.included), (b.state_inv, b.included_inv))
            i = rand(rng, 1:b.width)
            v = arr[i, j]
            v > smin || continue
            nv = (reset && v >= il) ? il - one(eltype(arr)) : v - one(eltype(arr))
            arr[i, j] = nv
            if v >= il && nv < il
                n = (i - 1) >> 6 + 1
                mask[n, j] &= ~(one(UInt64) << ((i - 1) & 63))
                changed = true
            end
        end
    end
    changed && TMCore.recount!(b, j)
end

function update_class_local!(m, ci, x, positive, rng, reset)
    p = m.params
    Tt, LFv, Lv, sv = p.T, p.LF, p.L, p.s
    v = clamp(score(m, ci, x), -Tt, Tt)
    upd = (positive ? (Tt - v) : (Tt + v)) / (2Tt)
    tI  = positive ? m.positive[ci] : m.negative[ci]
    tII = positive ? m.negative[ci] : m.positive[ci]
    @inbounds for j in 1:tI.nclauses
        rand(rng) < upd || continue
        n = Int(tI.count[j])
        vote = clause_vote(tI, j, x, LFv, m.ceiling, m.misscost)
        act = type_i_action(m.feedback, vote, ceiling(m.ceiling, n, LFv), rng)
        if act == FEEDBACK_REINFORCE
            feedback!(TypeIa(), tI, j, x, reinforce_allowed(m.budget, n, Lv),
                      promotion_room(m.budget, n, Lv, TypeIa()))
        elseif act == FEEDBACK_ERODE
            erode!(tI, j, sv, rng, reset)
        end
    end
    @inbounds for j in 1:tII.nclauses
        rand(rng) < upd || continue
        n = Int(tII.count[j])
        vote = clause_vote(tII, j, x, LFv, m.ceiling, m.misscost)
        if reject_branch(m.feedback, vote, ceiling(m.ceiling, n, LFv), rng)
            feedback!(TypeII(), tII, j, x, promotion_room(m.budget, n, Lv, TypeII()))
        end
    end
end

function train_local!(m, X, Y, rng, reset)
    for i in randperm(rng, length(Y))
        yi = findfirst(==(Y[i]), m.classes)
        update_class_local!(m, yi, X[i], true, rng, reset)
        for ci in 1:length(m.classes)
            ci == yi || update_class_local!(m, ci, X[i], false, rng, reset)
        end
    end
end

function load_sparse(path, width)
    X, Y = TMInput[], Bool[]
    for line in eachline(path)
        parts = split(line)
        v = falses(width)
        @inbounds for k in 2:length(parts)
            v[parse(Int, parts[k])] = true
        end
        push!(X, TMInput(v)); push!(Y, parts[1] == "1")
    end
    return X, Y
end

vocab = readlines(joinpath(DATA, "vocab.txt"))
WIDTH = length(vocab)
Xtr, Ytr = load_sparse(joinpath(DATA, "train.txt"), WIDTH)
Xte, Yte = load_sparse(joinpath(DATA, "test.txt"), WIDTH)

println("="^92)
println("Can a confidence gradient make an unreadable IMDb clause readable?")
println("="^92)
@printf("%d features, %d train, %d test, T %d S %d L %d LF %d, %d epochs\n\n",
        WIDTH, length(Xtr), length(Xte), T, S, L, LF, EPOCHS)

m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF, budget=PartialFreeze())
function train_epochs!(m, Xtr, Ytr, Xte, Yte, epochs)
    rng = MersenneTwister(20260908)
    best = 0.0
    for _ in 1:epochs
        train_local!(m, Xtr, Ytr, rng, true)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
    end
    return best
end

best = train_epochs!(m, Xtr, Ytr, Xte, Yte, EPOCHS)
@printf("accuracy %.4f  (published reference 0.9015; imdb-readability got 0.9012 without reset)\n", best)

# Did a gradient actually form under this configuration?
allstates = Int[]
for banks in (m.positive, m.negative), b in banks, j in 1:b.nclauses, i in 1:b.width
    b.state[i, j] >= b.include_limit && push!(allstates, Int(b.state[i, j]))
    b.state_inv[i, j] >= b.include_limit && push!(allstates, Int(b.state_inv[i, j]))
end
sort!(allstates)
@printf("included automata: median %d, max %d, %.1f%% above the threshold\n\n",
        allstates[end÷2], allstates[end], 100count(>(128), allstates) / length(allstates))

# --------------------------------------------------------------------------
# Rank a clause's literals, truncate, and measure what the truncation keeps.
# --------------------------------------------------------------------------
isin(mask, j, i) = !iszero(mask[(i - 1) >> 6 + 1, j] & (one(UInt64) << ((i - 1) & 63)))

"Every included literal of clause `j`, with its automaton state and satisfaction frequency."
function literals_of(b, j, X, LFv)
    lits = Tuple{Int,Bool,Int,Int}[]     # (position, negated, state, satisfied count)
    sat = zeros(UInt64, b.nchunks); persat = zeros(Int, b.width); fired = 0
    for x in X
        clause_vote!(sat, b, j, x, LFv, LiteralCapped()) == 0 && continue
        fired += 1
        @inbounds for n in 1:b.nchunks
            w = sat[n]; base = (n - 1) * 64
            while w != 0
                persat[base + trailing_zeros(w) + 1] += 1
                w &= w - one(UInt64)
            end
        end
    end
    for i in 1:b.width
        if isin(b.included, j, i)
            push!(lits, (i, false, Int(b.state[i, j]), persat[i]))
        elseif isin(b.included_inv, j, i)
            push!(lits, (i, true, Int(b.state_inv[i, j]), persat[i]))
        end
    end
    return lits, fired
end

matches(x, lits) = all(t -> t[2] ? !x[t[1]] : x[t[1]], lits)

function evaluate(lits, X, Y, cls)
    hits = findall(k -> matches(X[k], lits), eachindex(X))
    isempty(hits) && return (0, NaN)
    return (length(hits), count(k -> Y[k] == cls, hits) / length(hits))
end

println("="^92)
println("TRUNCATION: does a short rule keep the full rule's precision?")
println("="^92)
println("Ranking by automaton state (needs the gradient) vs by satisfaction frequency (does not).")
println()

for (ci, cls) in enumerate(m.classes), (pol, bank) in ((1, m.positive[ci]), (2, m.negative[ci]))
    for j in 1:bank.nclauses
        lits, fired = literals_of(bank, j, Xte, LF)
        length(lits) < 50 && continue
        nfull, pfull = evaluate(lits, Xte, Yte, cls)
        @printf("\n--- %s sentiment, %s clause %d: %d literals, fires %d/%d ---\n",
                cls ? "POSITIVE" : "NEGATIVE", pol == 1 ? "for" : "against", j,
                length(lits), fired, length(Xte))
        @printf("    full rule: matches %d docs, P(class) = %.3f\n", nfull, pfull)
        by_state = sort(lits, by = t -> -t[3])
        by_freq  = sort(lits, by = t -> -t[4])
        @printf("    %-6s %-28s %-28s\n", "top-N", "ranked by CONFIDENCE", "ranked by FREQUENCY")
        for N in (5, 10, 20, 50, 200)
            N > length(lits) && continue
            ns, ps = evaluate(by_state[1:N], Xte, Yte, cls)
            nf, pf = evaluate(by_freq[1:N], Xte, Yte, cls)
            @printf("    %-6d matches %5d  P %.3f      matches %5d  P %.3f\n", N, ns, ps, nf, pf)
        end
        top = by_state[1:min(12, end)]
        println("    top 12 by confidence:")
        for t in top
            @printf("      %s\"%s\"  (state %d, satisfied %d/%d)\n",
                    t[2] ? "NOT " : "", vocab[t[1]], t[3], t[4], fired)
        end
    end
end
