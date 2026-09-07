# Is a fuzzy clause really k rules plus a tail, or is it diffuse?
#
# FPTM keeps a clause's include set but breaks the identity between that set and the rule the clause
# encodes. A clause with 100 literals and LF = 50 fires on any 50 of them: an m-of-n rule, which is a
# much weaker interpretability class than a conjunction (prior art: Towell & Shavlik, KBANN). The
# include set is still printable, and a human still learns nothing from reading it.
#
# The cheap diagnostics already ran and both said to continue. Fuzziness is load-bearing (91.6% of
# nonzero votes strictly interior), and per-literal satisfaction frequency has a wide spread (median
# 0.638), so the mask is not uniform noise. What neither shows is *joint* structure: whether the
# literals that co-occur in a firing form a small number of recurring patterns.
#
# Method. For every clause, on every input where it votes above zero, record which included literals
# actually matched — the satisfied mask. That yields an empirical distribution over subsets of the
# include set. Then ask how concentrated it is.
#
# Pass/fail, stated in advance, from the project's own framing:
#   Low-entropy, concentrated masks -> the compression is real and recoverable, and the
#     interpretability story is repairable: this one clause is k rules plus a tail.
#   Diffuse masks -> FPTM trades the TM family's central selling point for efficiency, which the
#     field should know, because rules are the reason to prefer a TM over a small MLP.
#
# Operationally: if the top 10 distinct masks cover a majority of a clause's firings, it decomposes.
# If they cover a few percent, it does not.
#
#   julia --project=. research/mask-mining/run.jl [n_examples]

include(joinpath(@__DIR__, "..", "upstream.jl"))
include(joinpath(@__DIR__, "..", "mnist.jl"))
using Printf, Statistics
using TMCore

const NEX = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10_000
const WIDTH = 784

const FPTM_PATH = upstream("BooBSD-FuzzyPatternTM")
include(joinpath(FPTM_PATH, "src", "FuzzyPatternTM.jl"))
const FPTM = Main.FuzzyPatternTM

println("="^80)
println("Satisfied-mask mining on Hnilov's published 40-clause MNIST model")
println("="^80)

px, y_test, n_img, _, _ = mnist_test(joinpath(REPO_ROOT, "data"))
raw, trans = booleanize_both(px, n_img)
pub = FPTM.load(joinpath(FPTM_PATH, "models", "tm_optimized_40_fp.tm"))
acc(bv) = sum(FPTM.predict(pub, [FPTM.TMInput(b) for b in bv[1:2000]]) .== y_test[1:2000]) / 2000
bits = acc(raw) >= acc(trans) ? raw : trans
@printf("model loaded, booleanization verified at %.4f accuracy\n", max(acc(raw), acc(trans)))

X = [TMInput(Vector{Bool}(b)) for b in bits[1:min(NEX, n_img)]]
LF = pub.LF
@printf("mining %d examples, LF = %d\n\n", length(X), LF)

# Import every clause slot into TMCore banks so the packed satisfied mask is available.
banks = Tuple{Int,Int,ClauseBank}[]
for (ci, (_, ta)) in enumerate(sort(collect(pub.clauses), by = first))
    for (pol, (l, i)) in enumerate(((ta.positive_included_literals, ta.positive_included_literals_inverted),
                                    (ta.negative_included_literals, ta.negative_included_literals_inverted)))
        push!(banks, (ci, pol, ClauseBank(WIDTH, l, i)))
    end
end

"""
Mine one clause: hash each firing's satisfied mask and count occurrences. Hashing rather than storing
masks keeps memory flat — a clause with thousands of distinct masks would otherwise cost more to
analyse than to train.
"""
function mine(b::ClauseBank, j::Integer, X, LF)
    counts = Dict{UInt64,Int}()
    sat = zeros(UInt64, b.nchunks)
    firings = 0
    persat = zeros(Int, b.width)
    for x in X
        v = clause_vote!(sat, b, j, x, LF, LiteralCapped())
        v == 0 && continue
        firings += 1
        counts[hash(sat)] = get(counts, hash(sat), 0) + 1
        @inbounds for n in 1:b.nchunks
            w = sat[n]
            base = (n - 1) * 64
            while w != 0
                persat[base + trailing_zeros(w) + 1] += 1
                w &= w - one(UInt64)
            end
        end
    end
    return counts, firings, persat
end

results = NamedTuple[]
for (ci, pol, b) in banks, j in 1:b.nclauses
    counts, firings, persat = mine(b, j, X, LF)
    firings == 0 && continue
    n_lit = Int(b.count[j])
    freqs = sort(collect(values(counts)); rev = true)
    p = freqs ./ firings
    H = -sum(x -> x * log(x), p)
    Hnorm = firings > 1 ? H / log(firings) : 0.0
    core = count(>=(0.95 * firings), persat)          # satisfied in essentially every firing
    push!(results, (ci=ci, pol=pol, j=j, n_lit=n_lit, firings=firings,
                    distinct=length(counts),
                    top1=freqs[1] / firings,
                    top10=sum(freqs[1:min(10, end)]) / firings,
                    Hnorm=Hnorm, core=core))
end

med(f) = median([f(r) for r in results])
@printf("%d clause slots fired at least once, of %d\n\n", length(results), sum(b.nclauses for (_,_,b) in banks))
println("MEDIAN OVER CLAUSES")
println("-"^80)
@printf("  included literals            %6.1f\n", med(r -> r.n_lit))
@printf("  firings (of %d)           %6.1f\n", length(X), med(r -> r.firings))
@printf("  distinct satisfied masks     %6.1f\n", med(r -> r.distinct))
@printf("  distinct / firings           %6.3f   (1.0 = every firing a different mask)\n",
        med(r -> r.distinct / r.firings))
@printf("  top-1 mask coverage          %6.3f\n", med(r -> r.top1))
@printf("  top-10 mask coverage         %6.3f   <- the decomposition criterion\n", med(r -> r.top10))
@printf("  normalised entropy           %6.3f   (0 = one mask, 1 = all distinct)\n", med(r -> r.Hnorm))
@printf("  core literals (>=95%% of firings) %3.1f of %.1f included\n",
        med(r -> r.core), med(r -> r.n_lit))

println()
# Normalised entropy is H / log(firings), which is biased upward when firings are few: a clause that
# fired 8 times cannot have more than 8 distinct masks and lands at H = 1 by arithmetic rather than
# by being diffuse. Rank only clauses with enough firings to distinguish the two.
const MINFIRE = 500
solid = filter(r -> r.firings >= MINFIRE, results)
@printf("ranking restricted to the %d clauses with >= %d firings
", length(solid), MINFIRE)

println()
println("MOST CONCENTRATED (lowest normalised entropy)")
println("-"^80)
for r in sort(solid, by = x -> x.Hnorm)[1:min(5, end)]
    @printf("  class %d %s clause %2d: %3d literals, %5d firings, %5d distinct, top-10 %.3f, core %2d, H %.3f
",
            r.ci - 1, r.pol == 1 ? "pos" : "neg", r.j, r.n_lit, r.firings, r.distinct, r.top10, r.core, r.Hnorm)
end

println()
println("MOST DIFFUSE (highest normalised entropy)")
println("-"^80)
for r in sort(solid, by = x -> -x.Hnorm)[1:min(5, end)]
    @printf("  class %d %s clause %2d: %3d literals, %5d firings, %5d distinct, top-10 %.3f, core %2d, H %.3f
",
            r.ci - 1, r.pol == 1 ? "pos" : "neg", r.j, r.n_lit, r.firings, r.distinct, r.top10, r.core, r.Hnorm)
end

println()
top10s = [r.top10 for r in solid]
@printf("among those: median top-10 coverage %.3f, %.0f%% above 0.5, %.0f%% below 0.05
",
        median(top10s), 100count(>=(0.5), top10s) / length(top10s),
        100count(<(0.05), top10s) / length(top10s))

# ---------------------------------------------------------------------------
# Is the extracted core an actual rule?
# ---------------------------------------------------------------------------
# The core is the set of literals satisfied in essentially every firing. If it is a meaningful
# conjunction, then evaluating it *strictly* on its own should select the clause's class far above
# the 10% base rate. If it selects everything, the core is trivia and the decomposition is cosmetic.

"Literals satisfied in at least `frac` of this clause's firings, as a standalone strict conjunction."
function core_rule(b::ClauseBank, j::Integer, X, LF, frac)
    _, firings, persat = mine(b, j, X, LF)
    firings == 0 && return Int[], Int[]
    keep = findall(>=(frac * firings), persat)
    pos = [i for i in keep if !iszero(b.included[(i - 1) >> 6 + 1, j] & (one(UInt64) << ((i - 1) & 63)))]
    neg = [i for i in keep if !iszero(b.included_inv[(i - 1) >> 6 + 1, j] & (one(UInt64) << ((i - 1) & 63)))]
    return pos, neg
end

matches(bits, pos, neg) = all(i -> bits[i], pos) && all(i -> !bits[i], neg)

println()
println("IS THE CORE AN ACTUAL RULE?  (strict conjunction, evaluated standalone)")
println("-"^80)
# Polarity matters here and it is easy to misread. A *negative* clause votes AGAINST its class, so
# its core firing on that class far BELOW the 10% base rate is the rule working, not failing. Lift is
# therefore reported directionally: above 1 is correct for a positive clause, below 1 for a negative
# one, and 1.0 means the core carries no class information either way.
println("  clause                 core  fires   P(class)   lift    reads as")
labels = y_test[1:length(X)]
bitsv = bits[1:length(X)]

function report(r)
    b = banks[findfirst(t -> t[1] == r.ci && t[2] == r.pol, banks)][3]
    pos, neg = core_rule(b, r.j, X, LF, 0.95)
    (isempty(pos) && isempty(neg)) && return
    hits = findall(k -> matches(bitsv[k], pos, neg), 1:length(X))
    isempty(hits) && return
    prec = count(k -> labels[k] == r.ci - 1, hits) / length(hits)
    base = count(==(r.ci - 1), labels) / length(labels)
    lift = prec / base
    wanted = r.pol == 1 ? lift > 1.5 : lift < 0.67
    @printf("  class %d %s clause %2d  %3d  %5d     %.3f   %5.2fx   %s
",
            r.ci - 1, r.pol == 1 ? "pos" : "neg", r.j, length(pos) + length(neg),
            length(hits), prec, lift, wanted ? "rule holds" : "no signal")
end

for pol in (1, 2)
    println(pol == 1 ? "  -- positive clauses (core should select the class) --" :
                       "  -- negative clauses (core should avoid the class) --")
    for r in sort(filter(x -> x.pol == pol, solid), by = x -> x.Hnorm)[1:min(4, end)]
        report(r)
    end
end

# Eight hand-picked clauses are an anecdote. Run the same test over every clause with enough firings
# and report the distribution, which is the claim that can actually be defended.
println()
println("CORE-AS-RULE OVER ALL " * string(length(solid)) * " CLAUSES WITH >= " * string(MINFIRE) * " FIRINGS")
println("-"^80)
function core_lift(r)
    b = banks[findfirst(t -> t[1] == r.ci && t[2] == r.pol, banks)][3]
    pos, neg = core_rule(b, r.j, X, LF, 0.95)
    (isempty(pos) && isempty(neg)) && return nothing
    hits = findall(k -> matches(bitsv[k], pos, neg), 1:length(X))
    isempty(hits) && return nothing
    prec = count(k -> labels[k] == r.ci - 1, hits) / length(hits)
    base = count(==(r.ci - 1), labels) / length(labels)
    return (pol=r.pol, core=length(pos) + length(neg), n_lit=r.n_lit,
            fires=length(hits), lift=prec / base)
end
lifts = filter(!isnothing, [core_lift(r) for r in solid])
for pol in (1, 2)
    v = filter(x -> x.pol == pol, lifts)
    isempty(v) && continue
    l = sort([x.lift for x in v])
    held = pol == 1 ? count(>(1.5), l) : count(<(0.67), l)
    @printf("  %s clauses (n=%d): median lift %.2fx, quartiles %.2f / %.2f, core %.0f of %.0f literals
",
            pol == 1 ? "positive" : "negative", length(v), l[end÷2], l[max(1,end÷4)], l[min(end,3end÷4)],
            median([x.core for x in v]), median([x.n_lit for x in v]))
    @printf("    core carries the clause's class signal in %d of %d (%.0f%%)
",
            held, length(v), 100held / length(v))
end
