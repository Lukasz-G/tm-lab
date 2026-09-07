# The naive functions below are transcribed directly from FuzzyPatternTM's `feedback!`, operating on
# plain arrays with no packing. They are the specification; the packed implementations must agree
# with them state for state. Two entirely different mechanisms computing the same transition is a
# real check, the same way the evaluator differential is.

function naive_type_ia!(c, ci, x, il, smin, smax, reinforce)
    if reinforce
        for i in eachindex(x)
            x[i]  && c[i]  < smax && (c[i]  += 1)
            !x[i] && ci[i] < smax && (ci[i] += 1)
        end
    end
    for i in eachindex(x)                       # erosion is never gated by the budget
        !x[i] && c[i]  < il && c[i]  > smin && (c[i]  -= 1)
        x[i]  && ci[i] < il && ci[i] > smin && (ci[i] -= 1)
    end
end

function naive_type_ii!(c, ci, x, il)
    for i in eachindex(x)
        !x[i] && c[i]  < il && (c[i]  += 1)
        x[i]  && ci[i] < il && (ci[i] += 1)
    end
end

setstate!(bank, j, c, ci) = (bank.state[:, j] .= c; bank.state_inv[:, j] .= ci; nothing)

"Rebuild include masks and the cached count from the raw states."
function sync!(bank, j)
    for n in 1:bank.nchunks
        m = mi = zero(UInt64)
        for k in 0:TMCore.last_bit(bank, n)
            i = (n - 1) * 64 + k + 1
            bank.state[i, j]     >= bank.include_limit && (m  |= one(UInt64) << k)
            bank.state_inv[i, j] >= bank.include_limit && (mi |= one(UInt64) << k)
        end
        bank.included[n, j], bank.included_inv[n, j] = m, mi
    end
    TMCore.recount!(bank, j)
end

@testset "feedback matches the reference rule, state for state" begin
    rng = MersenneTwister(20260907)
    for _ in 1:60
        width = rand(rng, (17, 64, 65, 100, 200))
        bank = ClauseBank{UInt8}(width, 2; states = 256, include_limit = 128)
        il, smin, smax = bank.include_limit, bank.state_min, bank.state_max

        # States spanning excluded, just-included and saturated automata.
        c  = rand(rng, UInt8.(120:135), width)
        ci = rand(rng, UInt8.(120:135), width)
        v  = rand(rng, Bool, width)
        x  = TMInput(v)

        # Type Ia, reinforcement permitted.
        setstate!(bank, 1, c, ci); sync!(bank, 1)
        want_c, want_ci = copy(c), copy(ci)
        naive_type_ia!(want_c, want_ci, v, il, smin, smax, true)
        feedback!(TypeIa(), bank, 1, x, true)
        @test bank.state[:, 1] == want_c
        @test bank.state_inv[:, 1] == want_ci

        # Type Ia with the gate shut: erosion only, and already-included literals frozen too.
        setstate!(bank, 1, c, ci); sync!(bank, 1)
        want_c, want_ci = copy(c), copy(ci)
        naive_type_ia!(want_c, want_ci, v, il, smin, smax, false)
        feedback!(TypeIa(), bank, 1, x, false)
        @test bank.state[:, 1] == want_c
        @test bank.state_inv[:, 1] == want_ci

        # Type II.
        setstate!(bank, 1, c, ci); sync!(bank, 1)
        want_c, want_ci = copy(c), copy(ci)
        naive_type_ii!(want_c, want_ci, v, il)
        feedback!(TypeII(), bank, 1, x)
        @test bank.state[:, 1] == want_c
        @test bank.state_inv[:, 1] == want_ci

        # Masks and cached count must track the states after every rule, or the evaluator reads
        # stale data and nothing downstream is trustworthy.
        for n in 1:bank.nchunks, k in 0:TMCore.last_bit(bank, n)
            i = (n - 1) * 64 + k + 1
            @test !iszero(bank.included[n, 1] & (one(UInt64) << k)) == (bank.state[i, 1] >= il)
            @test !iszero(bank.included_inv[n, 1] & (one(UInt64) << k)) == (bank.state_inv[i, 1] >= il)
        end
        @test bank.count[1] == count(>=(il), bank.state[:, 1]) + count(>=(il), bank.state_inv[:, 1])
    end
end

@testset "literal budget policies" begin
    @test reinforce_allowed(GrowthGate(), 10, 10) == true
    @test reinforce_allowed(GrowthGate(), 11, 10) == false
    @test reinforce_allowed(HardCap(), 999, 10) == true
    @test promotion_room(GrowthGate(), 0, 10) == typemax(Int)
    @test promotion_room(HardCap(), 4, 10) == 6
    @test promotion_room(HardCap(), 12, 10) == 0

    # HardCap genuinely bounds clause size where GrowthGate does not: one Type Ia pass on an
    # all-ones input promotes every plain literal at once, which is the overshoot mechanism.
    width, L = 200, 10
    for (policy, bounded) in ((GrowthGate(), false), (HardCap(), true))
        bank = ClauseBank{UInt8}(width, 1; include_limit = 128)
        for _ in 1:3
            n = Int(bank.count[1])
            feedback!(TypeIa(), bank, 1, TMInput(trues(width)),
                      reinforce_allowed(policy, n, L), promotion_room(policy, n, L))
        end
        bounded ? (@test bank.count[1] <= L) : (@test bank.count[1] > L)
    end
end

@testset "TypeIb erodes and can drop literals" begin
    rng = MersenneTwister(7)
    bank = ClauseBank{UInt8}(128, 1; include_limit = 128)
    bank.state[:, 1] .= 0x80                        # every plain literal exactly at the threshold
    sync!(bank, 1)
    @test bank.count[1] == 128
    feedback!(TypeIb(), bank, 1, 64, rng)
    @test bank.count[1] < 128                       # some eroded out of the include set
    # Positions are drawn with replacement, as in the reference, so one may be hit more than once.
    # What is exact is the total: 64 draws, every automaton above the floor, so 64 decrements.
    @test all(bank.state[:, 1] .<= 0x80)
    @test sum(0x80 .- Int.(bank.state[:, 1])) == 64

    # Automata already at the floor cannot go lower.
    bank2 = ClauseBank{UInt8}(64, 1; include_limit = 128)
    bank2.state[:, 1] .= 0x00
    bank2.state_inv[:, 1] .= 0x00
    sync!(bank2, 1)
    feedback!(TypeIb(), bank2, 1, 100, rng)
    @test all(bank2.state[:, 1] .== 0x00)
    @test bank2.count[1] == 0
end
