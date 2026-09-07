# What a failed literal costs.
#
# FPTM charges exactly 1 per failed literal, regardless of how strongly the model believes that
# literal belongs in the clause. But that belief is already in the model: a Tsetlin automaton sitting
# just over the include threshold is a tentative inclusion, one near `state_max` is a confident one,
# and evaluation throws the distinction away.
#
# The idea costs no new hyperparameter — the include band's midpoint is derived from
# `include_limit` and `state_max` — and it keeps votes integral, which matters because `T`, the vote
# histogram and every downstream comparison assume integers.
#
# It is not free, though, and the cost is structural rather than incidental: weighting by automaton
# state means evaluation needs the automata, so an inference-only model (which drops them, and is 8x
# smaller for it) cannot score under this policy. That is a real trade and the reason the uniform
# policy stays the default regardless of how the measurement comes out.

# `MissCostPolicy` and `UniformMissCost` are declared in misscost_fwd.jl, ahead of ClauseBank, so
# that clause_vote can name the default.

"""
    ConfidenceWeightedMissCost(threshold = 0)

A failed literal costs 2 if its automaton is at or above `threshold`, 1 otherwise.

`threshold = 0` means "use the midpoint of the nominal include band", which is the obvious choice and
a **bad** one. Measured on a reference-trained model: `include_limit` is 128 and `state_max` 255, so
the nominal midpoint is 191, but included automata actually occupy 128 to 167 with a median of 136.
Nothing ever reaches 191 and the policy silently degenerates to the uniform one.

The reason is the growth gate. It freezes *all* reinforcement once a clause exceeds `L`, not just new
promotions, so included automata stop climbing almost immediately and cluster just above the include
threshold. Confidence, the quantity this policy wants to exploit, barely accumulates under the
reference training dynamics. (Under `HardCap`, which never freezes reinforcement, the same automata
saturate at 255 — so the distribution is a property of the budget policy, not of the data.)

Use [`calibrate_misscost!`](@ref) to set a threshold from the model's own state distribution.

The reasoning: a literal the model is confident about is a stronger claim, so its failure is stronger
evidence against the clause. A literal barely over the threshold is a tentative claim and its failure
is weak evidence.

Two consequences worth stating plainly. Confident misses effectively halve `LF` for the clauses that
hold them, so the same `LF` buys less tolerance — any comparison against the uniform policy is
therefore not holding "amount of fuzziness" fixed. And scoring now requires the automata, so
inference-only models cannot use this policy.
"""
struct ConfidenceWeightedMissCost <: MissCostPolicy
    threshold::UInt16
end
ConfidenceWeightedMissCost() = ConfidenceWeightedMissCost(UInt16(0))

"""
    confidence_threshold(policy, bank)

Automaton state at or above which an included literal counts as confidently included. A policy
threshold of 0 falls back to the nominal midpoint of the include band.
"""
@inline confidence_threshold(p::ConfidenceWeightedMissCost, b::ClauseBank) =
    p.threshold == 0 ? (b.include_limit + (b.state_max - b.include_limit) ÷ 2) : p.threshold

"""
    miss_cost(policy, bank, j, x) -> Int

Total cost of the literals clause `j` fails on example `x`.
"""
@inline function miss_cost(::UniformMissCost, b::ClauseBank, j::Integer, x::TMInput)
    total = 0
    inc, inv, ch = b.included, b.included_inv, x.chunks
    @inbounds @simd for n in 1:b.nchunks
        total += count_ones(miss_mask(inc[n, j], inv[n, j], ch[n]))
    end
    return total
end

function miss_cost(p::ConfidenceWeightedMissCost, b::ClauseBank{S}, j::Integer, x::TMInput) where {S}
    st, sti = b.state, b.state_inv
    st === nothing && throw(ArgumentError(
        "ConfidenceWeightedMissCost needs the Tsetlin automata, but this model was loaded " *
        "inference-only. Save with include_states = true, or score with UniformMissCost."))
    mid = confidence_threshold(p, b)
    total = 0
    inc, inv, ch = b.included, b.included_inv, x.chunks
    @inbounds for n in 1:b.nchunks
        w = miss_mask(inc[n, j], inv[n, j], ch[n])
        base = (n - 1) * 64
        # Walk only the failed positions: cost is proportional to misses, not to width.
        while w != 0
            i = base + trailing_zeros(w) + 1
            bit = one(UInt64) << (i - base - 1)
            confident = (!iszero(inc[n, j] & bit) && st[i, j] >= mid) ||
                        (!iszero(inv[n, j] & bit) && sti[i, j] >= mid)
            total += confident ? 2 : 1
            w &= w - one(UInt64)
        end
    end
    return total
end
