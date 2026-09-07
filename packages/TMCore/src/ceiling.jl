# The clause-vote ceiling.
#
# A fuzzy clause's vote counts down from a ceiling, one per failed literal, clipped at zero. The
# ceiling is the *unit* of the vote: it sets whether short general clauses vote as loudly as long
# specific ones, whether L and LF are independent knobs, how T calibrates, and whether votes are
# comparable across clauses at all.
#
# The sources disagree, so it is a policy rather than a constant.
#
#   FPTM paper (arXiv:2508.08350 sec.2)   n == 0 ? LF : min(n, LF)
#   FuzzyPatternTM, Hnilov's reference    0 < n < LF ? n : LF          -- identical
#   Tsetlin.jl, Hnilov's optimized fork   LF                           -- flat
#
# Two of three agree, so `LiteralCapped` is the default. `FlatLF` is kept because it is what the
# fast implementation actually does, and reproducing its published numbers requires being able to
# reproduce its semantics. On a converged model the two differ on well under 1% of clauses, but
# "small at convergence" is not "absent during training".

"""
    CeilingPolicy

How a clause's maximum vote is derived from its included-literal count `n` and the hyperparameter
`LF`. See [`LiteralCapped`](@ref) and [`FlatLF`](@ref).
"""
abstract type CeilingPolicy end

"""
    LiteralCapped()

The FPTM paper's rule, matching Hnilov's reference implementation: a clause cannot vote more
strongly than it has literals to justify. The empty clause is special-cased *upward* to `LF`, which
preserves the classical "empty clause is true" convention — `min` alone would send it to zero.
"""
struct LiteralCapped <: CeilingPolicy end

"""
    FlatLF()

Every clause ceilings at `LF` regardless of how many literals it includes, as Tsetlin.jl does.
Cheaper — it needs no per-clause literal count at evaluation time — and over-votes any clause
holding between 1 and `LF-1` literals.
"""
struct FlatLF <: CeilingPolicy end

"""
    ceiling(policy, n, LF)

Maximum vote for a clause including `n` literals.
"""
@inline ceiling(::LiteralCapped, n::Integer, LF::Integer) = ifelse(n == 0, LF, min(n, LF))
@inline ceiling(::FlatLF, ::Integer, LF::Integer) = LF

"True when `policy` never consults the literal count, so maintaining it is not required for voting."
needs_literal_count(::LiteralCapped) = true
needs_literal_count(::FlatLF) = false
