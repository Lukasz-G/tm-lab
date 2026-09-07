# Measurement hook.
#
# The evaluator already computes the satisfied mask and normally discards it; this exposes it at
# model level so the interpretability work needs nothing from the algorithm beyond what is here.
#
# This is not speculative instrumentation. FPTM keeps a clause's include set but breaks the identity
# between that set and the rule the clause encodes: a clause with 100 literals and LF = 50 fires on
# any 50 of them, which is an m-of-n rule and a much weaker interpretability class than a
# conjunction. On Hnilov's published MNIST model, 91.6% of nonzero clause votes are strictly
# interior, so partial matching is the normal case rather than an edge case, and *which* literals
# carried the match is the only route back to something a human can read.

"""
    ClauseStats

Per-clause-slot observations over a dataset.

- `votes[v+1, k]` — how often clause slot `k` voted exactly `v`.
- `satisfied[i, k]` — how often literal position `i` was satisfied in slot `k`. Positions the clause
  does not include are always zero.
- `counts[k]`, `ceilings[k]` — the slot's included-literal count and its resulting vote ceiling.
- `slot_of[k]` — `(class_index, polarity, clause_index)`, polarity 1 positive and 2 negative.
"""
struct ClauseStats
    nexamples::Int
    LF::Int
    votes::Matrix{Int}
    satisfied::Matrix{Int}
    counts::Vector{Int}
    ceilings::Vector{Int}
    slot_of::Vector{Tuple{Int,Int,Int}}
end

"""
    observe(model, X) -> ClauseStats

Evaluate every clause on every example, recording the vote distribution and the satisfied mask.

Costs one extra AND per chunk over plain prediction, plus unpacking the mask into per-position
counters, which dominates. Intended for analysis rather than for the training loop.
"""
function observe(m::TMClassifier, X::AbstractVector{TMInput})
    LF = m.params.LF
    width = m.params.width
    banks = Tuple{Int,Int,ClauseBank}[]
    for ci in eachindex(m.classes)
        push!(banks, (ci, 1, m.positive[ci]))
        push!(banks, (ci, 2, m.negative[ci]))
    end
    nslots = sum(b.nclauses for (_, _, b) in banks)

    votes = zeros(Int, LF + 1, nslots)
    satisfied = zeros(Int, width, nslots)
    counts = zeros(Int, nslots)
    ceilings = zeros(Int, nslots)
    slot_of = Vector{Tuple{Int,Int,Int}}(undef, nslots)

    k = 0
    mask = UInt64[]
    for (ci, pol, b) in banks
        length(mask) == b.nchunks || (mask = zeros(UInt64, b.nchunks))
        for j in 1:b.nclauses
            k += 1
            slot_of[k] = (ci, pol, j)
            counts[k] = Int(b.count[j])
            ceilings[k] = ceiling(m.ceiling, counts[k], LF)
            for x in X
                v = clause_vote!(mask, b, j, x, LF, m.ceiling)
                votes[v + 1, k] += 1
                @inbounds for n in 1:b.nchunks
                    w = mask[n]
                    base = (n - 1) * 64
                    while w != 0
                        satisfied[base + trailing_zeros(w) + 1, k] += 1
                        w &= w - one(UInt64)          # clear the lowest set bit
                    end
                end
            end
        end
    end
    return ClauseStats(length(X), LF, votes, satisfied, counts, ceilings, slot_of)
end

"""
    interior_fraction(stats) -> Vector{Float64}

Per slot, the share of *nonzero* votes that fall strictly below the ceiling.

This is the diagnostic that says whether fuzziness is load-bearing. Near zero means the clause
behaves like a strict conjunction and `LF` is decorative for it; near one means it is genuinely
operating as an m-of-n rule. Slots that never fired are reported as zero.
"""
function interior_fraction(s::ClauseStats)
    out = zeros(Float64, size(s.votes, 2))
    for k in axes(s.votes, 2)
        nz = sum(@view s.votes[2:end, k])
        nz == 0 && continue
        out[k] = (nz - s.votes[s.ceilings[k] + 1, k]) / nz
    end
    return out
end

"""
    satisfaction_spread(stats) -> Vector{Float64}

Per slot, the range (max minus min) of satisfaction frequency across the literals the clause
includes. Near zero means every included literal matches about equally often and the satisfied mask
carries no structure to mine; a wide spread means the clause has a near-always-on core plus a tail,
which is the decomposition that makes an m-of-n clause explicable.
"""
function satisfaction_spread(s::ClauseStats)
    out = zeros(Float64, size(s.satisfied, 2))
    for k in axes(s.satisfied, 2)
        s.counts[k] == 0 && continue
        lo, hi = typemax(Int), -1
        for i in axes(s.satisfied, 1)
            f = s.satisfied[i, k]
            # A position with zero satisfactions may simply not be included; only included
            # positions are meaningful, and those are exactly the ones that can be satisfied.
            f == 0 && continue
            lo = min(lo, f); hi = max(hi, f)
        end
        hi < 0 && continue
        out[k] = (hi - lo) / s.nexamples
    end
    return out
end

"""
    vote_histogram(stats) -> Vector{Int}

Vote counts pooled over every clause slot, indexed by vote value `0:LF`.
"""
vote_histogram(s::ClauseStats) = [sum(@view s.votes[v + 1, :]) for v in 0:s.LF]
