# Declared ahead of ClauseBank so that `clause_vote` can name a default miss-cost policy; the
# implementations live in misscost.jl, after ClauseBank exists.

"""
    MissCostPolicy

How much a failed literal subtracts from a clause's vote. FPTM charges exactly 1, discarding the
automaton confidence the model already holds. See [`UniformMissCost`](@ref) and
[`ConfidenceWeightedMissCost`](@ref).
"""
abstract type MissCostPolicy end

"""
    UniformMissCost()

Every failed literal costs 1, as FPTM specifies. Stays bit-parallel — one popcount per chunk — and
needs no automata, so inference-only models can score under it. The default.
"""
struct UniformMissCost <: MissCostPolicy end
