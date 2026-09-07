using Test
using TMCore

@testset "TMCore" begin

@testset "TMInput packing" begin
    for len in (1, 63, 64, 65, 127, 128, 200)
        v = rand(Bool, len)
        x = TMInput(v)
        @test length(x) == len
        @test collect(x) == v
        @test all(x[i] == v[i] for i in 1:len)
    end
    # Padding bits in the final chunk stay zero.
    x = TMInput(fill(true, 65))
    @test x.chunks[2] == one(UInt64)
    @test_throws BoundsError x[66]
end

@testset "ceiling policies" begin
    LF = 5
    # The empty clause is special-cased upward under both, preserving "empty clause is true".
    @test ceiling(LiteralCapped(), 0, LF) == LF
    @test ceiling(FlatLF(), 0, LF) == LF
    # They agree at or above LF...
    for n in LF:(LF + 10)
        @test ceiling(LiteralCapped(), n, LF) == ceiling(FlatLF(), n, LF) == LF
    end
    # ...and diverge strictly between 1 and LF-1, which is the whole disagreement.
    for n in 1:(LF - 1)
        @test ceiling(LiteralCapped(), n, LF) == n
        @test ceiling(FlatLF(), n, LF) == LF
        @test ceiling(FlatLF(), n, LF) > ceiling(LiteralCapped(), n, LF)
    end
end

@testset "clause evaluation" begin
    width = 100
    # clause 1: x3 ∧ x7 ∧ ¬x11      (3 literals)
    # clause 2: x1                   (1 literal, below LF -> ceiling differs by policy)
    # clause 3: empty                (votes the ceiling always)
    # clause 4: x5 ∧ ¬x5             (contradiction, can never be satisfied)
    bank = ClauseBank(width, [[3, 7], [1], Int[], [5]], [[11], Int[], Int[], [5]])
    @test length(bank) == 4
    @test bank.count == Int32[3, 1, 0, 2]
    # Clause 4 includes x5 and ¬x5 at the same position: two literals, one position. The count
    # follows FuzzyPatternTM and says two, since that is what feeds LiteralCapped.
    @test refresh_counts!(bank).count == Int32[3, 1, 0, 2]
    @test count_ones(bank.included[1, 4] | bank.included_inv[1, 4]) == 1   # positions, not literals

    LF = 5
    v = falses(width); v[3] = true; v[7] = true          # ¬x11 holds since v[11] is false
    x = TMInput(v)

    @test clause_vote(bank, 1, x, LF, LiteralCapped()) == 3    # ceiling 3, no misses
    @test clause_vote(bank, 1, x, LF, FlatLF()) == 5           # ceiling 5, no misses
    @test clause_vote(bank, 3, x, LF, LiteralCapped()) == LF   # empty clause
    # A contradiction holds two literals and exactly one of them fails on any input, whatever the
    # input is — so it votes ceiling(2) - 1 = 1, never reaching its ceiling. The packed kernel
    # produces exactly one miss bit for the position, matching FuzzyPatternTM's two-list loop.
    @test clause_vote(bank, 4, x, LF, LiteralCapped()) == 1
    @test clause_vote(bank, 4, TMInput(trues(width)), LF, LiteralCapped()) == 1
    @test clause_vote(bank, 4, TMInput(falses(width)), LF, LiteralCapped()) == 1
    @test clause_vote(bank, 4, x, LF, FlatLF()) == LF - 1

    # One literal short.
    v2 = copy(v); v2[7] = false
    x2 = TMInput(v2)
    @test clause_vote(bank, 1, x2, LF, LiteralCapped()) == 2
    @test clause_vote(bank, 1, x2, LF, FlatLF()) == 4

    # Enough misses drives it to zero and it clips there rather than going negative.
    v3 = falses(width); v3[11] = true                     # all three literals fail
    @test clause_vote(bank, 1, TMInput(v3), LF, LiteralCapped()) == 0
    @test clause_vote(bank, 1, TMInput(v3), 2, FlatLF()) == 0
end

@testset "satisfied mask" begin
    width = 70                                            # spans two chunks
    bank = ClauseBank(width, [[3, 65]], [[11]])
    v = falses(width); v[3] = true; v[65] = true
    x = TMInput(v)
    sat = zeros(UInt64, bank.nchunks)

    @test clause_vote!(sat, bank, 1, x, 5, LiteralCapped()) == 3
    @test sat[1] == (one(UInt64) << 2) | (one(UInt64) << 10)   # x3 and ¬x11 both matched
    @test sat[2] == one(UInt64)                                 # x65 matched

    # Drop x65: the vote falls by one and that bit leaves the mask.
    v[65] = false
    @test clause_vote!(sat, bank, 1, TMInput(v), 5, LiteralCapped()) == 2
    @test sat[2] == zero(UInt64)
    @test sat[1] == (one(UInt64) << 2) | (one(UInt64) << 10)

    # The mask never reports a position that is not included.
    for n in 1:bank.nchunks
        @test sat[n] & ~(bank.included[n, 1] | bank.included_inv[n, 1]) == 0
    end
    @test_throws DimensionMismatch clause_vote!(zeros(UInt64, 1), bank, 1, TMInput(v), 5)
end

@testset "vote equals ceiling minus misses, exhaustively" begin
    # Brute-force agreement with the definition, over random clauses and inputs.
    width = 40
    for _ in 1:200
        lits = sort(unique(rand(1:width, rand(0:6))))
        invs = sort(unique(setdiff(rand(1:width, rand(0:6)), lits)))
        bank = ClauseBank(width, [lits], [invs])
        v = rand(Bool, width)
        x = TMInput(v)
        misses = count(i -> !v[i], lits) + count(i -> v[i], invs)
        n = length(lits) + length(invs)
        for LF in (1, 3, 5), pol in (LiteralCapped(), FlatLF())
            @test clause_vote(bank, 1, x, LF, pol) == max(0, ceiling(pol, n, LF) - misses)
        end
    end
end

@testset "bank_vote" begin
    bank = ClauseBank(64, [[1], [2], Int[]], [Int[], Int[], Int[]])
    v = falses(64); v[1] = true
    x = TMInput(v)
    # clause 1 matches (1), clause 2 misses (0), clause 3 is empty (ceiling 4).
    @test bank_vote(bank, x, 4, LiteralCapped()) == 1 + 0 + 4
end

@testset "hyperparameter guards" begin
    h = Hyperparameters(T = 10, S = 125, L = 10, LF = 5, width = 784)
    @test h.s == round(Int, 784 / 125)
    @test occursin("s=", string(h))

    # LF = 0 is the silent killer: nothing votes, nothing learns, every margin is zero.
    err = try Hyperparameters(T = 10, S = 125, L = 10, LF = 0, width = 784) catch e; e end
    @test err isa ArgumentError
    @test occursin("cannot learn", err.msg)
    @test occursin("LF = 1", err.msg)

    @test_throws ArgumentError Hyperparameters(T = 0, S = 1, L = 1, LF = 1, width = 8)
    @test_throws ArgumentError Hyperparameters(T = 1, S = 0, L = 1, LF = 1, width = 8)

    @test check_transfer(h, 784) === h
    @test (@test_logs (:warn,) check_transfer(h, 1568)) === h
end

@testset "symbol-encoding guard" begin
    # H = 1: a clause one bit short matches every symbol in the vocabulary.
    @test alias_count(256, 1, 100) ≈ 99.0
    # H = 2 at D = 32 is the configuration that motivated the guard: ~12 aliasing symbols.
    @test 12 < alias_count(32, 2, 100) < 13
    # H >= 4 makes the concern vanish.
    @test alias_count(256, 4, 100) < 1e-3
    @test alias_count(256, 8, 100) < 1e-8
    # Alias count grows linearly in vocabulary size.
    @test alias_count(256, 2, 1000) ≈ 10 * alias_count(256, 2, 100) rtol = 0.02

    @test check_symbol_encoding(256, 8, 100) < 1e-8
    @test_throws ArgumentError check_symbol_encoding(32, 2, 100)
    @test_throws ArgumentError check_symbol_encoding(256, 1, 100)
    @test_throws ArgumentError alias_count(32, 33, 10)
end

end
