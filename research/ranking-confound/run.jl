# Does the clause contribute anything, or is discriminative ranking just recovering the globally
# best features?
#
# discriminative-ranking claimed a readable rule extracted from a 3,443-literal clause. But the score
# used — P(satisfied | in class) - P(satisfied | out of class) — is essentially chi-square feature
# relevance, and the 12,800 IMDb features were **already** selected by chi-square against the label in
# the preparation step. So the top-ranked literals may simply be the globally most discriminative
# n-grams, which would make the clause irrelevant and the result hollow.
#
# Three rankings, same scoring function, evaluated identically:
#
#   CLAUSE   rank only the literals the clause includes            (what was claimed)
#   GLOBAL   rank all 12,800 features, ignoring the model entirely (the null hypothesis)
#   RANDOM   a random subset of features, scored and ranked        (floor)
#
# If GLOBAL matches or beats CLAUSE, the clause contributed nothing and the earlier claim needs
# withdrawing. If CLAUSE beats GLOBAL, the model is selecting which discriminative features to
# combine, and the explanation is about the clause after all.
#
# Also reported: how much the two top-10 sets overlap. Identical sets would settle it immediately.
#
#   julia --project=. research/ranking-confound/run.jl [epochs]

using Printf, Random, Statistics
using TMCore

const DATA = normpath(joinpath(@__DIR__, "..", "..", "data", "imdb"))
const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
const T, S, L, LF = 18, 1000, 64, 64

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
println("Confound check: does the clause matter, or is this just global feature relevance?")
println("="^94)
@printf("%d features, %d train, %d test, %d epochs\n\n", WIDTH, length(Xtr), length(Xte), EPOCHS)

function train_model(epochs)
    m = TMClassifier(Ytr, WIDTH; clauses_per_class=2, T=T, S=S, L=L, LF=LF)
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
@printf("model accuracy %.4f\n\n", best)

isin(mask, j, i) = !iszero(mask[(i - 1) >> 6 + 1, j] & (one(UInt64) << ((i - 1) & 63)))
sat(x, t) = t[2] ? !x[t[1]] : x[t[1]]
matches(x, lits) = all(t -> sat(x, t), lits)

function evaluate(lits, X, Y, cls)
    hits = findall(k -> matches(X[k], lits), eachindex(X))
    isempty(hits) && return (0, NaN)
    return (length(hits), count(k -> Y[k] == cls, hits) / length(hits))
end

"""
Discriminative score for a candidate literal, on train only. Computed identically for every ranking
so the three arms differ solely in which literals are *candidates*.
"""
function score_literals(cands, X, Y, cls)
    inmask = [y == cls for y in Y]
    nin, nout = count(inmask), count(!, inmask)
    s = Float64[]
    for t in cands
        si = so = 0
        @inbounds for k in eachindex(X)
            if sat(X[k], t)
                inmask[k] ? (si += 1) : (so += 1)
            end
        end
        push!(s, si / nin - so / nout)
    end
    return s
end

rng = MersenneTwister(11)
for (ci, cls) in enumerate(m.classes)
    bank = m.positive[ci]
    j = 1
    clause_lits = Tuple{Int,Bool}[]
    for i in 1:bank.width
        isin(bank.included, j, i) && push!(clause_lits, (i, false))
        isin(bank.included_inv, j, i) && push!(clause_lits, (i, true))
    end
    length(clause_lits) < 100 && continue

    # GLOBAL considers every feature in both polarities, with no reference to the model at all.
    global_lits = vcat([(i, false) for i in 1:WIDTH], [(i, true) for i in 1:WIDTH])
    rand_lits = [(rand(rng, 1:WIDTH), rand(rng, Bool)) for _ in 1:length(clause_lits)]

    @printf("\n=== %s sentiment, positive clause (%d literals of %d possible) ===\n",
            cls ? "POSITIVE" : "NEGATIVE", length(clause_lits), 2WIDTH)

    sc = score_literals(clause_lits, Xtr, Ytr, cls)
    sg = score_literals(global_lits, Xtr, Ytr, cls)
    sr = score_literals(rand_lits, Xtr, Ytr, cls)
    by_c = clause_lits[sortperm(sc, rev=true)]
    by_g = global_lits[sortperm(sg, rev=true)]
    by_r = rand_lits[sortperm(sr, rev=true)]

    @printf("    %-5s %-24s %-24s %-24s\n", "N", "CLAUSE literals", "GLOBAL (model ignored)", "RANDOM subset")
    for N in (3, 5, 10, 20)
        nc, pc = evaluate(by_c[1:N], Xte, Yte, cls)
        ng, pg = evaluate(by_g[1:N], Xte, Yte, cls)
        nr, pr = evaluate(by_r[1:N], Xte, Yte, cls)
        @printf("    %-5d %6d docs  P %.3f    %6d docs  P %.3f    %6d docs  P %.3f\n",
                N, nc, pc, ng, pg, nr, pr)
    end

    ov = length(intersect(Set(by_c[1:10]), Set(by_g[1:10])))
    @printf("    top-10 overlap between CLAUSE and GLOBAL: %d of 10\n", ov)
    println("    GLOBAL top 10 (no model involved):")
    for k in 1:10
        t = by_g[k]
        @printf("      %-4s\"%s\"\n", t[2] ? "NOT " : "", vocab[t[1]])
    end
end

println()
println("If GLOBAL matches CLAUSE, the model contributed nothing and the earlier result was hollow.")
