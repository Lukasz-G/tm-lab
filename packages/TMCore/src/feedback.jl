# Feedback rules, as swappable components parameterized by type.
#
# Semantics follow FuzzyPatternTM, Hnilov's reference implementation, which is also what the FPTM
# paper describes. Each rule updates the Tsetlin automata for one clause and rebuilds that clause's
# include mask in the same pass, so the packed masks the evaluator reads are never stale.
#
# The automaton array is one byte per literal position, so this loop is inherently O(width) and not
# bit-parallel — unlike evaluation. That asymmetry is why FPTM is fast at inference and ordinary at
# training.
#
# Portions derived from Tsetlin.jl (rebuilding the include mask inside the state-update pass rather
# than as a second sweep). Copyright (c) 2024-2026 Artem Hnilov. MIT License. See NOTICE.md.

"""
    FeedbackRule

A rule that mutates clause automata in response to one example. Dispatching on the rule type rather
than branching keeps strict/fuzzy, classification/regression and future variants separable at no
runtime cost.
"""
abstract type FeedbackRule end

"""
    TypeIa()

The clause fired. Reinforce every literal that agrees with the input, and erode the *excluded*
literals that disagree with it, making the clause a tighter fit to this example.

**Deterministic** — this is precisely where FPTM departs from the classical TM, which applies these
increments with probability `1/s`. Removing that randomness is what makes FPTM's optimal `S` roughly
the square of the classical TM's.

The literal-budget policy supplies two things. `reinforce` says whether the increment half runs at
all — under `GrowthGate` a clause already over `L` freezes entirely, including its already-included
literals. `room` rations how many literals may newly cross the include threshold. Erosion is never
gated by either.
"""
struct TypeIa <: FeedbackRule end

"""
    TypeIb()

The clause did not fire. Erode `s` randomly chosen automata from each polarity, which is how unused
clauses decay and are recycled. The only randomness in FPTM's feedback path.
"""
struct TypeIb <: FeedbackRule end

"""
    TypeII()

Applied to clauses that fired on an example of the *wrong* class. Push excluded literals that
contradict the input toward inclusion, so the clause becomes specific enough to stop matching.
Only excluded automata move: a literal already included is left alone.
"""
struct TypeII <: FeedbackRule end

# Recompute the cached literal count for one clause from its masks.
@inline function recount!(b::ClauseBank, j::Integer)
    c = 0
    @inbounds for n in 1:b.nchunks
        c += count_ones(b.included[n, j]) + count_ones(b.included_inv[n, j])
    end
    @inbounds b.count[j] = Int32(c)
    return c
end

"""
    feedback!(TypeIa(), bank, j, x, reinforce, room)

Reinforce agreeing literals and erode disagreeing excluded ones. Returns the new literal count.
"""
function feedback!(::TypeIa, b::ClauseBank{S}, j::Integer, x::TMInput,
                   reinforce::Bool, room::Integer=typemax(Int)) where {S}
    st, sti = b.state::Matrix{S}, b.state_inv::Matrix{S}
    il, smin, smax = b.include_limit, b.state_min, b.state_max
    room = Int(room)
    @inbounds for n in 1:b.nchunks
        chunk = x.chunks[n]
        base = (n - 1) * 64
        stop = last_bit(b, n)
        m, mi = b.included[n, j], b.included_inv[n, j]
        for k in 0:stop
            i = base + k + 1
            bit = one(UInt64) << k
            on = !iszero(chunk & bit)
            s0, s1 = st[i, j], sti[i, j]

            # Plain literal xᵢ agrees when the bit is set; the negated literal when it is not.
            if on
                if reinforce && s0 < smax && (s0 >= il || room > 0)
                    s0 += one(S)
                    s0 == il && (room -= 1)          # this one crossed into the include set
                end
                if s1 < il && s1 > smin
                    s1 -= one(S)
                end
            else
                if reinforce && s1 < smax && (s1 >= il || room > 0)
                    s1 += one(S)
                    s1 == il && (room -= 1)
                end
                if s0 < il && s0 > smin
                    s0 -= one(S)
                end
            end

            st[i, j], sti[i, j] = s0, s1
            m  = ifelse(s0 >= il, m | bit,  m & ~bit)
            mi = ifelse(s1 >= il, mi | bit, mi & ~bit)
        end
        b.included[n, j], b.included_inv[n, j] = m, mi
    end
    return recount!(b, j)
end

"""
    feedback!(TypeIb(), bank, j, s, rng)

Erode `s` random automata from each polarity. Only automata that are already at `state_min` are
unaffected; a literal eroded below `include_limit` leaves the clause.
"""
function feedback!(::TypeIb, b::ClauseBank{S}, j::Integer, s::Integer, rng=Random.default_rng()) where {S}
    st, sti = b.state::Matrix{S}, b.state_inv::Matrix{S}
    il, smin = b.include_limit, b.state_min
    changed = false
    @inbounds for _ in 1:s
        for (arr, mask) in ((st, b.included), (sti, b.included_inv))
            i = rand(rng, 1:b.width)
            v = arr[i, j]
            if v > smin
                v -= one(S)
                arr[i, j] = v
                if v == il - one(S)                  # dropped out of the include set
                    n = (i - 1) >> 6 + 1
                    mask[n, j] &= ~(one(UInt64) << ((i - 1) & 63))
                    changed = true
                end
            end
        end
    end
    return changed ? recount!(b, j) : Int(b.count[j])
end

"""
    feedback!(TypeII(), bank, j, x, room = typemax(Int))

Push excluded literals that contradict `x` toward inclusion. Included literals are untouched.

`room` rations how many literals may cross into the include set. **Neither reference gates Type II
by `L` at all** — the budget check appears only in the Type Ia path — so a clause can grow here with
no budget involvement whatsoever. That is a second, independent reason `L` fails to bound clause
size, on top of the growth-gate behaviour. `GrowthGate` therefore passes an unrestricted `room` and
reproduces the references exactly; `HardCap` passes real headroom, because a policy that claims to
be a cap has to hold on every path that grows a clause.
"""
function feedback!(::TypeII, b::ClauseBank{S}, j::Integer, x::TMInput,
                   room::Integer=typemax(Int)) where {S}
    st, sti = b.state::Matrix{S}, b.state_inv::Matrix{S}
    il = b.include_limit
    room = Int(room)
    @inbounds for n in 1:b.nchunks
        chunk = x.chunks[n]
        base = (n - 1) * 64
        stop = last_bit(b, n)
        m, mi = b.included[n, j], b.included_inv[n, j]
        for k in 0:stop
            i = base + k + 1
            bit = one(UInt64) << k
            on = !iszero(chunk & bit)
            s0, s1 = st[i, j], sti[i, j]
            # Requiring xᵢ = 1 would falsify the clause on an example where it is 0, and vice versa.
            if !on && s0 < il
                if s0 < il - one(S) || room > 0
                    s0 += one(S)
                    s0 == il && (room -= 1)
                end
            elseif on && s1 < il
                if s1 < il - one(S) || room > 0
                    s1 += one(S)
                    s1 == il && (room -= 1)
                end
            end
            st[i, j], sti[i, j] = s0, s1
            m  = ifelse(s0 >= il, m | bit,  m & ~bit)
            mi = ifelse(s1 >= il, mi | bit, mi & ~bit)
        end
        b.included[n, j], b.included_inv[n, j] = m, mi
    end
    return recount!(b, j)
end
