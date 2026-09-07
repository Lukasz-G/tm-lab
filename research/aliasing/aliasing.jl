# Bit- versus symbol-granular fuzziness over a sparse distributed code.
#
# Question: when FPTM's fuzzy clause vote is applied to GraphTM's hypervector-encoded symbols,
# does tolerating a missing *bit* still mean "matched a sub-pattern", or does it mean "matched a
# corrupted symbol that aliases to every other symbol sharing those bits"?
#
# This is arithmetic, not a training run. See README.md for the model and the verdict.

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# D  hypervector size, bits
# H  hypervector_bits, bits set per symbol; each symbol is a uniformly random H-subset of [D]
# V  vocabulary size, distinct symbols
# m  symbols superposed (OR-ed) into one node's feature vector
# k  symbols a clause conjoins; the clause includes k*H positive literals
#
# Assumptions, stated so they can be attacked:
#   A1  symbol codes are independent uniform H-subsets (GraphTM assigns them randomly)
#   A2  a node's feature vector is the bitwise OR of the codes of the symbols present
#   A3  bits are treated as independent when computing union statistics; exact for the
#       hypergeometric pair calculation below, an approximation only where noted (D >> H)

"P(a given bit is set) after OR-ing m independent symbol codes."
p_set(D, H, m) = 1 - (1 - H / D)^m

"P(|B_s ∩ B_s'| == j) for two independent random H-subsets of [D]. Hypergeometric, exact."
function p_overlap(D, H, j)
    0 <= j <= H || return 0.0
    H <= D || error("H > D")
    num = binomial(big(H), big(j)) * binomial(big(D - H), big(H - j))
    den = binomial(big(D), big(H))
    return Float64(num / den)
end

"""
Expected number of OTHER symbols in a vocabulary of V whose code overlaps a given symbol's in at
least H-f bits — i.e. the symbols that, on their own, drive a clause targeting `s` to a vote within
`f` of full under bit-granular fuzziness. f = 0 counts exact code collisions.
"""
function alias_count(D, H, V, f)
    tail = sum(p_overlap(D, H, j) for j in (H-f):H)
    self = p_overlap(D, H, H)          # the j == H term includes s' being code-identical to s
    return (V - 1) * tail, (V - 1) * self
end

"P(at most `budget` of `n` independent bits fail), each failing with probability `u`."
function p_at_most(n, u, budget)
    budget >= n && return 1.0
    u <= 0 && return 1.0
    u >= 1 && return 0.0
    t = (1 - u)^n                       # i = 0
    acc = t
    for i in 1:budget
        t *= (n - i + 1) / i * u / (1 - u)
        acc += t
    end
    return min(acc, 1.0)
end

"False-positive rate for a k-symbol clause against a node holding m unrelated symbols."
function fp_bit(D, H, m, k, lf_bits)
    u = 1 - p_set(D, H, m)              # P(a required bit is absent)
    return p_at_most(k * H, u, lf_bits)
end

function fp_symbol(D, H, m, k, lf_symbols)
    q = p_set(D, H, m)^H                # P(one whole symbol is spuriously present)
    return p_at_most(k, 1 - q, lf_symbols)
end

using Printf
fmt(x) = x == 0 ? "0" :
         x >= 1e4 ? @sprintf("%.3g", x) :
         x >= 0.01 ? @sprintf("%.4f", x) : @sprintf("%.1e", x)

# ---------------------------------------------------------------------------
# Table 1 — alias set size. How many OTHER symbols individually trigger a
# near-full vote against a clause that learned symbol s?
# ---------------------------------------------------------------------------
println("="^78)
println("TABLE 1  Alias set size, bit-granular fuzziness, V = 100 symbols")
println("          N(f) = expected number of other symbols that alone drive the vote")
println("          to within f of full. N(0) = exact code collisions (a design bug if > 0).")
println("="^78)
println(rpad("D", 6), rpad("H", 4), rpad("N(0) collisions", 18), rpad("N(1)", 12), rpad("N(2)", 12))
println("-"^78)
for D in (32, 128, 256, 512, 1024), H in (1, 2, 4, 8, 16)
    H > D && continue
    n1, n0 = alias_count(D, H, 100, 1)
    n2, _ = alias_count(D, H, 100, 2)
    println(rpad(D, 6), rpad(H, 4), rpad(fmt(n0), 18), rpad(fmt(n1), 12), rpad(fmt(n2), 12))
end

# ---------------------------------------------------------------------------
# Table 2 — does it degrade with vocabulary? The stated pass/fail.
# ---------------------------------------------------------------------------
println()
println("="^78)
println("TABLE 2  Alias set growth with vocabulary, at fixed H (D = 256)")
println("         Pass/fail: bit-granular degrades as V grows; symbol-granular does not.")
println("="^78)
println(rpad("H", 4), rpad("V", 8), rpad("bit-gran N(1)", 16), rpad("symbol-gran N(0)", 20))
println("-"^78)
for H in (2, 4, 8), V in (10, 100, 1_000, 10_000)
    n1, n0 = alias_count(256, H, V, 1)
    println(rpad(H, 4), rpad(V, 8), rpad(fmt(n1), 16), rpad(fmt(n0), 20))
end

# ---------------------------------------------------------------------------
# Table 3 — head to head at an operating point. One clause, k symbols, against
# a node holding m unrelated symbols. Equal tolerance budgets are not equal:
# "1 bit" and "1 symbol" are different amounts of forgiveness by construction,
# so both are shown.
# ---------------------------------------------------------------------------
println()
println("="^78)
println("TABLE 3  False-positive rate, k = 3 symbols per clause, D = 256")
println("         strict = no tolerance. bit LF=1,2 = that many missing bits allowed.")
println("         sym LF=1 = one whole symbol allowed to be absent.")
println("="^78)
println(rpad("H", 4), rpad("m", 4), rpad("strict", 12), rpad("bit LF=1", 12),
        rpad("bit LF=2", 12), rpad("sym LF=1", 12))
println("-"^78)
for H in (2, 4, 8), m in (1, 4, 16)
    st = fp_bit(256, H, m, 3, 0)
    b1 = fp_bit(256, H, m, 3, 1)
    b2 = fp_bit(256, H, m, 3, 2)
    s1 = fp_symbol(256, H, m, 3, 1)
    println(rpad(H, 4), rpad(m, 4), rpad(fmt(st), 12), rpad(fmt(b1), 12),
            rpad(fmt(b2), 12), rpad(fmt(s1), 12))
end

# ---------------------------------------------------------------------------
# Table 4 — the inflation factor, isolated. How much does allowing one missing
# bit multiply the false-positive rate, relative to strict?
# ---------------------------------------------------------------------------
println()
println("="^78)
println("TABLE 4  False-positive inflation from allowing ONE missing bit (k = 3, D = 256)")
println("         ratio = fp(bit LF=1) / fp(strict). 1.0 would mean fuzziness is free.")
println("="^78)
println(rpad("H", 4), rpad("m", 4), rpad("P(bit set)", 14), rpad("inflation x", 14))
println("-"^78)
for H in (2, 4, 8, 16), m in (1, 4, 16)
    st = fp_bit(256, H, m, 3, 0)
    b1 = fp_bit(256, H, m, 3, 1)
    println(rpad(H, 4), rpad(m, 4), rpad(fmt(p_set(256, H, m)), 14),
            rpad(fmt(b1 / st), 14))
end

println()
println("Closed form for Table 4: inflation = 1 + n*(1-p)/p for n = k*H required bits,")
println("p = P(bit set). Sparse codes make p small, so (1-p)/p is large and the penalty")
println("for one forgiven bit is worst exactly where the encoding is most distributed.")
