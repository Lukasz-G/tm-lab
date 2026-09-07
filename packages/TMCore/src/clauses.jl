# Bit-packed clause bank and the fuzzy clause evaluator.
#
# Portions derived from Tsetlin.jl: the packed include-mask representation and the branch-free miss
# kernel `((lit ⊻ lit_inv) & chunk) ⊻ lit`, whose popcount is the number of failed literals.
# Copyright (c) 2024-2026 Artem Hnilov. MIT License. See NOTICE.md.

"""
    ClauseBank{S}

One polarity's worth of clauses over a fixed input width, bit-packed.

`included[n, j]` holds, for clause `j` and chunk `n`, the mask of positions whose plain literal is
included; `included_inv` the same for negated literals. `state`/`state_inv` are the Tsetlin automata
those masks are thresholded from, and are `nothing` for an inference-only model.

`count[j]` caches the number of included literals in clause `j`. It is maintained rather than
recomputed because [`LiteralCapped`](@ref) consults it on every evaluation; recomputing it would
turn an O(chunks) inner loop into two.
"""
struct ClauseBank{S<:Unsigned}
    width::Int                              # input bits
    nchunks::Int
    nclauses::Int
    included::Matrix{UInt64}                # nchunks × nclauses
    included_inv::Matrix{UInt64}
    state::Union{Matrix{S},Nothing}         # width × nclauses, or nothing when inference-only
    state_inv::Union{Matrix{S},Nothing}
    count::Vector{Int32}
end

"""
    ClauseBank(width, literals, literals_inverted; S = UInt8)

Build an inference-only bank from per-clause lists of 1-based literal positions. This is the entry
point for importing a model trained elsewhere.
"""
function ClauseBank(width::Integer,
                    literals::AbstractVector{<:AbstractVector{<:Integer}},
                    literals_inverted::AbstractVector{<:AbstractVector{<:Integer}};
                    S::Type{<:Unsigned}=UInt8)
    length(literals) == length(literals_inverted) ||
        throw(ArgumentError("literal and inverted-literal clause counts differ"))
    nclauses = length(literals)
    nch = cld(width, 64)
    inc = zeros(UInt64, nch, nclauses)
    inv = zeros(UInt64, nch, nclauses)
    cnt = zeros(Int32, nclauses)
    for j in 1:nclauses
        for (mask, idxs) in ((inc, literals[j]), (inv, literals_inverted[j]))
            for i in idxs
                1 <= i <= width || throw(ArgumentError("literal index $i outside width $width"))
                mask[(i - 1) >> 6 + 1, j] |= one(UInt64) << ((i - 1) & 63)
            end
        end
        cnt[j] = Int32(length(literals[j]) + length(literals_inverted[j]))
    end
    return ClauseBank{S}(Int(width), nch, nclauses, inc, inv, nothing, nothing, cnt)
end

Base.length(b::ClauseBank) = b.nclauses

"""
    refresh_counts!(bank)

Recompute the cached literal counts from the include masks.

Counts **literals, not positions**: a clause that includes both `x` and `¬x` at the same position
holds two literals, so the count is `count_ones(inc) + count_ones(inv)` rather than
`count_ones(inc | inv)`. This matters because the count feeds [`LiteralCapped`](@ref), and
FuzzyPatternTM — which defines that ceiling — stores literals as two index lists and sums their
lengths, so it counts such a clause as two. Tsetlin.jl's `include_literals_sum` ORs instead and
would say one, but it only ever uses that figure for the `L` growth gate, never for a ceiling, so
the two are not actually in conflict. Contradictory clauses are rare, and permanently unsatisfiable
when they occur, but the ceiling has to agree with the reference or the two are not the same model.
"""
function refresh_counts!(b::ClauseBank)
    @inbounds for j in 1:b.nclauses
        c = 0
        for n in 1:b.nchunks
            c += count_ones(b.included[n, j]) + count_ones(b.included_inv[n, j])
        end
        b.count[j] = Int32(c)
    end
    return b
end

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
#
# Miss kernel, per chunk and per bit position:
#
#   included  inverted   meaning                    val bit
#   --------  --------   ------------------------   ---------------------------
#      0         0       literal not included       0            (never a miss)
#      1         0       x must be 1                1 iff x == 0
#      0         1       x must be 0                1 iff x == 1
#      1         1       contradiction              1            (always a miss)
#
# `val = ((inc ⊻ inv) & chunk) ⊻ inc` realises all four without a branch, and `count_ones(val)` is
# the number of failed literals in the chunk.

@inline miss_mask(inc::UInt64, inv::UInt64, chunk::UInt64) = (((inc ⊻ inv) & chunk) ⊻ inc)

"""
    clause_vote(bank, j, x, LF, policy) -> Int

Fuzzy vote of clause `j` on example `x`: the ceiling less one per failed literal, clipped at zero.
"""
@inline function clause_vote(b::ClauseBank, j::Integer, x::TMInput, LF::Integer,
                             policy::CeilingPolicy=LiteralCapped())
    @boundscheck (1 <= j <= b.nclauses && x.len == b.width) || throw(BoundsError(b, j))
    inc, inv, ch = b.included, b.included_inv, x.chunks
    misses = 0
    @inbounds @simd for n in 1:b.nchunks
        misses += count_ones(miss_mask(inc[n, j], inv[n, j], ch[n]))
    end
    return max(0, ceiling(policy, Int(@inbounds b.count[j]), Int(LF)) - misses)
end

"""
    clause_vote!(satisfied, bank, j, x, LF, policy) -> Int

As [`clause_vote`](@ref), but also writes the **satisfied mask** into `satisfied`: the included
literal positions that actually matched, packed the same way as the include masks.

This is the measurement hook. The miss mask is computed anyway, so recording costs one extra AND
per chunk; nothing here is reconstructed after the fact. Recovering *which* literals carried a
partial match is what turns an m-of-n clause back into something inspectable.
"""
@inline function clause_vote!(satisfied::AbstractVector{UInt64}, b::ClauseBank, j::Integer,
                              x::TMInput, LF::Integer, policy::CeilingPolicy=LiteralCapped())
    @boundscheck begin
        (1 <= j <= b.nclauses && x.len == b.width) || throw(BoundsError(b, j))
        length(satisfied) == b.nchunks || throw(DimensionMismatch("satisfied buffer must hold $(b.nchunks) chunks"))
    end
    inc, inv, ch = b.included, b.included_inv, x.chunks
    misses = 0
    @inbounds @simd for n in 1:b.nchunks
        i, v = inc[n, j], inv[n, j]
        m = miss_mask(i, v, ch[n])
        satisfied[n] = (i | v) & ~m
        misses += count_ones(m)
    end
    return max(0, ceiling(policy, Int(@inbounds b.count[j]), Int(LF)) - misses)
end

"""
    bank_vote(bank, x, LF, policy) -> Int

Summed vote of every clause in the bank.
"""
function bank_vote(b::ClauseBank, x::TMInput, LF::Integer, policy::CeilingPolicy=LiteralCapped())
    total = 0
    @inbounds for j in 1:b.nclauses
        total += clause_vote(b, j, x, LF, policy)
    end
    return total
end
