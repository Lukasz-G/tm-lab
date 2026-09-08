# Rank a clause's literals by how well they separate the classes.
#
# literal-ranking failed twice, and the second failure said why. Both signals it tried — automaton
# confidence and satisfaction frequency — measure *how often a literal's condition holds*, not *how
# much it distinguishes the classes*. An n-gram absent from nearly every review scores maximally on
# both and tells you nothing.
#
# The discriminative score asks the right question directly:
#
#     score(literal) = P(satisfied | in class) - P(satisfied | out of class)
#
# It needs no confidence gradient, so it applies to a stock FPTM trained exactly as published, and it
# can be computed from the training set alone.
#
# Protocol matters here and is easy to get wrong: literals are ranked on TRAIN and the truncated rule
# is evaluated on TEST. Ranking on the test set would be choosing the rule using the answers.
#
# Baselines: frequency (known useless, kept so the comparison is visible) and a random ranking, which
# is the floor any method has to clear.
#
# Run research/imdb-readability/prepare.py first.
#
#   julia --project=. research/discriminative-ranking/run.jl [epochs]

using Printf, Random, Statistics
using TMCore

const DATA = normpath(joinpath(@__DIR__, "..", "..", "data", "imdb"))
const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
const T, S, L, LF = 18, 1000, 64, 64
const CLAUSES = 2

function load_sparse(path, width)
    X, Y = TMInput[], Bool[]
    for line in eachline(path)
        p = split(line)
        v = falses(width)
        @inbounds for k in 2:length(p)
            v[parse(Int, p[k])] = true
        end
        push!(X, TMInput(v)); push!(Y, p[1] == "1")
    end
    return X, Y
end

vocab = readlines(joinpath(DATA, "vocab.txt"))
WIDTH = length(vocab)
Xtr, Ytr = load_sparse(joinpath(DATA, "train.txt"), WIDTH)
Xte, Yte = load_sparse(joinpath(DATA, "test.txt"), WIDTH)

println("="^94)
println("Discriminative literal ranking on IMDb — does a short rule survive?")
println("="^94)
@printf("%d features, %d train, %d test, T %d S %d L %d LF %d, %d epochs\n\n",
        WIDTH, length(Xtr), length(Xte), T, S, L, LF, EPOCHS)

function train_model(epochs)
    m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(20260908)
    best = 0.0
    for _ in 1:epochs
        train!(m, Xtr, Ytr; rng=rng)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
    end
    return m, best
end

m, best = train_model(EPOCHS)
@printf("stock FPTM, trained as published: accuracy %.4f\n\n", best)

isin(mask, j, i) = !iszero(mask[(i - 1) >> 6 + 1, j] & (one(UInt64) << ((i - 1) & 63)))

"Included literals of clause `j` as (position, negated)."
function literals_of(b, j)
    out = Tuple{Int,Bool}[]
    for i in 1:b.width
        isin(b.included, j, i) && push!(out, (i, false))
        isin(b.included_inv, j, i) && push!(out, (i, true))
    end
    return out
end

sat(x, t) = t[2] ? !x[t[1]] : x[t[1]]
matches(x, lits) = all(t -> sat(x, t), lits)

"""
Discriminative score per literal, computed on the training set only.

`P(satisfied | in class) - P(satisfied | out of class)`. A literal satisfied by everything scores
zero however common it is, which is exactly the failure mode that sank the frequency ranking.
"""
function discriminative_scores(lits, X, Y, cls)
    inmask = [y == cls for y in Y]
    nin, nout = count(inmask), count(!, inmask)
    scores = Float64[]
    for t in lits
        si = so = 0
        @inbounds for k in eachindex(X)
            if sat(X[k], t)
                inmask[k] ? (si += 1) : (so += 1)
            end
        end
        push!(scores, si / nin - so / nout)
    end
    return scores
end

function evaluate(lits, X, Y, cls)
    hits = findall(k -> matches(X[k], lits), eachindex(X))
    isempty(hits) && return (0, NaN)
    return (length(hits), count(k -> Y[k] == cls, hits) / length(hits))
end

function freq_scores(lits, X)
    [count(k -> sat(X[k], t), eachindex(X)) / length(X) for t in lits]
end

println("="^94)
println("TRUNCATION — ranked on TRAIN, evaluated on TEST")
println("="^94)

rng = MersenneTwister(7)
for (ci, cls) in enumerate(m.classes), (pol, bank) in ((1, m.positive[ci]), (2, m.negative[ci]))
    for j in 1:bank.nclauses
        lits = literals_of(bank, j)
        length(lits) < 100 && continue
        base = count(==(cls), Yte) / length(Yte)
        @printf("\n--- %s sentiment, %s clause %d: %d literals (base rate %.3f) ---\n",
                cls ? "POSITIVE" : "NEGATIVE", pol == 1 ? "for" : "against", j, length(lits), base)

        ds = discriminative_scores(lits, Xtr, Ytr, cls)
        fs = freq_scores(lits, Xtr)
        by_disc = lits[sortperm(ds, rev=true)]
        by_freq = lits[sortperm(fs, rev=true)]
        by_rand = shuffle(rng, copy(lits))

        @printf("    %-5s %-24s %-24s %-24s\n", "N", "DISCRIMINATIVE", "frequency", "random")
        for N in (3, 5, 10, 20, 50)
            N > length(lits) && continue
            nd, pd = evaluate(by_disc[1:N], Xte, Yte, cls)
            nf, pf = evaluate(by_freq[1:N], Xte, Yte, cls)
            nr, pr = evaluate(by_rand[1:N], Xte, Yte, cls)
            @printf("    %-5d %6d docs  P %.3f    %6d docs  P %.3f    %6d docs  P %.3f\n",
                    N, nd, pd, nf, pf, nr, pr)
        end

        println("    top 10 discriminative:")
        for k in 1:min(10, length(by_disc))
            t = by_disc[k]
            idx = findfirst(==(t), lits)
            @printf("      %-4s\"%s\"   (score %+.3f)\n", t[2] ? "NOT " : "", vocab[t[1]], ds[idx])
        end
    end
end
