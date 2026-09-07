# Are the extracted clause cores readable by a human?
#
# mask-mining showed that a fuzzy clause decomposes into a strict conjunctive core plus a tolerance
# tail, and that positive-clause cores are high-precision rules — 92% of them, median 7.4x lift. But
# it ran on MNIST, where a literal is a pixel. "Readable" there meant low-entropy and checkable, not
# meaningful. A human cannot look at pixel 412 and learn anything.
#
# IMDb is the honest test: literals are words and n-grams, so an extracted core either reads as a
# sentiment rule or it does not, and there is no room to flatter the result.
#
# Run research/imdb-readability/prepare.py first.
#
#   julia --project=. research/imdb-readability/run.jl [clauses_per_class] [epochs]

using Printf, Random, Statistics
using TMCore

const DATA = normpath(joinpath(@__DIR__, "..", "..", "data", "imdb"))
const CLAUSES = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const EPOCHS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30

# Published binary IMDb configuration, from FuzzyPatternTM's examples/IMDb/imdb_minimal.jl.
const T, S, L, LF = 18, 1000, 64, 64

function load_sparse(path, width)
    X, Y = TMInput[], Bool[]
    for line in eachline(path)
        parts = split(line)
        v = falses(width)
        @inbounds for k in 2:length(parts)
            v[parse(Int, parts[k])] = true
        end
        push!(X, TMInput(v))
        push!(Y, parts[1] == "1")
    end
    return X, Y
end

vocab = readlines(joinpath(DATA, "vocab.txt"))
WIDTH = length(vocab)
@printf("vocabulary: %d features\n", WIDTH)
Xtr, Ytr = load_sparse(joinpath(DATA, "train.txt"), WIDTH)
Xte, Yte = load_sparse(joinpath(DATA, "test.txt"), WIDTH)
@printf("documents: %d train, %d test\n\n", length(Xtr), length(Xte))

println("="^80)
@printf("IMDb, %d clauses/class, T %d, S %d, L %d, LF %d, %d epochs\n", CLAUSES, T, S, L, LF, EPOCHS)
println("="^80)

function train_loop(m, Xtr, Ytr, Xte, Yte, epochs)
    rng = MersenneTwister(20260908)
    best = 0.0
    for e in 1:epochs
        t = @elapsed train!(m, Xtr, Ytr; rng=rng)
        a = accuracy(predict(m, Xte), Yte)
        a > best && (best = a)
        @printf("epoch %2d  test %.4f  best %.4f  (%.0f s)\n", e, a, best, t)
    end
    return best
end

m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
best = train_loop(m, Xtr, Ytr, Xte, Yte, EPOCHS)
@printf("\nbest test accuracy %.4f   (published 1-clause-per-class reference: 0.9015)\n\n", best)

# ---------------------------------------------------------------------------
# Extract each clause's core and print it as words.
# ---------------------------------------------------------------------------
"Literals satisfied in at least `frac` of this clause's firings, split by polarity."
function core_of(b::ClauseBank, j::Integer, X, LF, frac)
    sat = zeros(UInt64, b.nchunks)
    persat = zeros(Int, b.width)
    firings = 0
    for x in X
        clause_vote!(sat, b, j, x, LF, LiteralCapped()) == 0 && continue
        firings += 1
        @inbounds for n in 1:b.nchunks
            w = sat[n]; base = (n - 1) * 64
            while w != 0
                persat[base + trailing_zeros(w) + 1] += 1
                w &= w - one(UInt64)
            end
        end
    end
    firings == 0 && return Int[], Int[], 0
    keep = findall(>=(frac * firings), persat)
    isin(mask, i) = !iszero(mask[(i - 1) >> 6 + 1, j] & (one(UInt64) << ((i - 1) & 63)))
    return [i for i in keep if isin(b.included, i)],
           [i for i in keep if isin(b.included_inv, i)], firings
end

matches(x, pos, neg) = all(i -> x[i], pos) && all(i -> !x[i], neg)

println("="^80)
println("CLAUSE CORES, AS WORDS")
println("="^80)
for (ci, cls) in enumerate(m.classes), (pol, bank) in ((1, m.positive[ci]), (2, m.negative[ci]))
    for j in 1:bank.nclauses
        pos, neg, firings = core_of(bank, j, Xte, LF, 0.95)
        (isempty(pos) && isempty(neg)) && continue
        hits = findall(k -> matches(Xte[k], pos, neg), eachindex(Xte))
        prec = isempty(hits) ? NaN : count(k -> Yte[k] == cls, hits) / length(hits)
        label = cls ? "POSITIVE sentiment" : "NEGATIVE sentiment"
        @printf("\n--- %s, %s clause %d ---\n", label, pol == 1 ? "for" : "against", j)
        @printf("    %d included literals, fired on %d of %d test docs\n",
                Int(bank.count[j]), firings, length(Xte))
        @printf("    core: %d literals, matches %d docs, P(%s) = %.3f  (base rate %.3f)\n",
                length(pos) + length(neg), length(hits), label, prec,
                count(==(cls), Yte) / length(Yte))
        if !isempty(pos)
            println("    REQUIRES PRESENT:")
            for i in first(pos, 25)
                println("      \"", vocab[i], "\"")
            end
            length(pos) > 25 && @printf("      ... and %d more\n", length(pos) - 25)
        end
        if !isempty(neg)
            println("    REQUIRES ABSENT:")
            for i in first(neg, 25)
                println("      \"", vocab[i], "\"")
            end
            length(neg) > 25 && @printf("      ... and %d more\n", length(neg) - 25)
        end
    end
end
