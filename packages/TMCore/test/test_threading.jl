using Test
using TMCore
using Random: MersenneTwister

# Threading is only worth having if it is provably deterministic, so these tests assert the property
# rather than the speedup. A timing test would be flaky on a loaded machine and would not catch the
# failure that matters, which is a race producing a different model.
#
# Run under `julia -t auto` for these to mean anything. At one thread they still pass, but they no
# longer distinguish a correct implementation from a racy one, so the suite says so out loud.

@testset "threading" begin

if Threads.nthreads() == 1
    @info "test_threading: running with 1 thread — determinism assertions still run but cannot " *
          "detect a race. Use `julia -t auto` to make them meaningful."
end

function toy(n, width, seed)
    rng = MersenneTwister(seed)
    X = [TMInput(rand(rng, Bool, width)) for _ in 1:n]
    # Three classes so that threading has something to spread, and a learnable rule so a broken
    # update shows up as accuracy rather than only as a mismatch.
    Y = [x[1] ? 1 : (x[2] ? 2 : 3) for x in X]
    return X, Y
end

"Every automaton state in the model, in a fixed order — the full fingerprint, not a summary."
function fingerprint(m)
    out = UInt64[]
    for banks in (m.positive, m.negative), b in banks
        append!(out, vec(b.included))
        append!(out, vec(b.included_inv))
        append!(out, UInt64.(vec(b.state)))
        append!(out, UInt64.(vec(b.state_inv)))
        append!(out, UInt64.(b.count))
    end
    return out
end

mk(Y, width) = TMClassifier(Y, width; clauses_per_class=20, T=8, S=20, L=10, LF=4)

@testset "threaded training is bit-identical to itself" begin
    X, Y = toy(400, 24, 7)
    a, b = mk(Y, 24), mk(Y, 24)
    for _ in 1:3
        train!(a, X, Y; rng=MersenneTwister(99), parallel=:classes)
        train!(b, X, Y; rng=MersenneTwister(99), parallel=:classes)
    end
    # This is the assertion the whole feature rests on: same seed, same model, whatever the
    # scheduler did with the tasks.
    @test fingerprint(a) == fingerprint(b)
end

@testset "threaded differs from serial, and only in the RNG stream" begin
    X, Y = toy(400, 24, 8)
    s, t = mk(Y, 24), mk(Y, 24)
    train!(s, X, Y; rng=MersenneTwister(5), parallel=:none)
    train!(t, X, Y; rng=MersenneTwister(5), parallel=:classes)
    # Documented, not accidental: independent per-class streams cannot reproduce one stream
    # interleaved across classes. If this ever passes, the threaded path is not using its own
    # streams and the determinism guarantee above is empty.
    @test fingerprint(s) != fingerprint(t)

    # Same algorithm, same distribution: both must actually learn, and neither may be systematically
    # worse than the other. A race would usually still satisfy the inequality above while failing
    # here. The bar is the majority-class rate (0.5 on this label distribution) rather than a round
    # number picked by eye — an absolute threshold just encodes how long this toy happened to train.
    Xe, Ye = toy(200, 24, 9)
    for _ in 1:15
        train!(s, X, Y; rng=MersenneTwister(1), parallel=:none)
        train!(t, X, Y; rng=MersenneTwister(1), parallel=:classes)
    end
    as, at = accuracy(predict(s, Xe), Ye), accuracy(predict(t, Xe), Ye)
    @test as > 0.65 && at > 0.65                 # majority class is 0.5
    @test abs(as - at) < 0.12                    # different draws, not a different algorithm
end

@testset "batch predict is bit-identical to the serial loop" begin
    X, Y = toy(300, 24, 11)
    m = mk(Y, 24)
    for _ in 1:5
        train!(m, X, Y; rng=MersenneTwister(3))
    end
    Xe, _ = toy(500, 24, 12)
    # predict writes nothing, so threading it cannot change a single output. Comparing against the
    # explicit per-example loop pins that rather than trusting it.
    @test predict(m, Xe) == [predict(m, x) for x in Xe]
    @test predict(m, TMInput[]) == Int[]
end

@testset "clause parallelism is deterministic, on both sides of the work gate" begin
    # `:clauses` decides once per epoch whether threading is worth it, and the two branches are
    # different code. Both have to be deterministic, so both are exercised here rather than whichever
    # one this machine's shape happens to select.
    #
    # The wide-input route to getting above the gate was abandoned deliberately. Padding a narrow
    # signal out to 4096 bits made the task unlearnable, and for an instructive reason: the padding
    # is constant, so every clause can include its negation for free, and those free literals fill
    # the `L` growth gate before the real signal gets in — the same effect `research/cifar-messages/`
    # measured. An arm that cannot learn says nothing about whether its threading is correct, so the
    # gate is crossed with clause count instead.
    Xs, Ys = toy(200, 24, 21)
    below = () -> mk(Ys, 24)
    @test !TMCore._worth_threading(below())
    a, b = below(), below()
    for _ in 1:3
        train!(a, Xs, Ys; rng=MersenneTwister(4), parallel=:clauses)
        train!(b, Xs, Ys; rng=MersenneTwister(4), parallel=:clauses)
    end
    @test fingerprint(a) == fingerprint(b)

    Xl, Yl = toy(200, 256, 22)
    above = () -> TMClassifier(Yl, 256; clauses_per_class=2800, T=12, S=16, L=10, LF=4)
    @test TMCore._worth_threading(above())
    c, d = above(), above()
    for _ in 1:3
        train!(c, Xl, Yl; rng=MersenneTwister(4), parallel=:clauses)
        train!(d, Xl, Yl; rng=MersenneTwister(4), parallel=:clauses)
    end
    @test fingerprint(c) == fingerprint(d)

    # Distinct from the other schedules, for the documented RNG-stream reason.
    e, f = above(), above()
    train!(e, Xl, Yl; rng=MersenneTwister(4), parallel=:none)
    train!(f, Xl, Yl; rng=MersenneTwister(4), parallel=:classes)
    @test fingerprint(e) != fingerprint(c)
    @test fingerprint(f) != fingerprint(c)

    # And it must still learn: a race would usually still satisfy the inequalities above.
    for _ in 1:5
        train!(c, Xl, Yl; rng=MersenneTwister(2), parallel=:clauses)
        train!(e, Xl, Yl; rng=MersenneTwister(2), parallel=:none)
    end
    ac, ae = accuracy(predict(c, Xl), Yl), accuracy(predict(e, Xl), Yl)
    @test ac > 0.9 && ae > 0.9
    @test abs(ac - ae) < 0.1              # different draws, not a different algorithm
end

@testset "batched scoring matches the serial score exactly" begin
    rng = MersenneTwister(31)
    X = [TMInput(rand(rng, Bool, 512)) for _ in 1:120]
    Y = [x[1] ? 1 : (x[2] ? 2 : 3) for x in X]
    m = TMClassifier(Y, 512; clauses_per_class=200, T=12, S=16, L=10, LF=4)
    train!(m, X, Y; rng=MersenneTwister(1))

    ncl = length(m.classes)
    half = m.positive[1].nclauses
    nchunk = TMCore._score_chunks(ncl, half)
    pv, nv = zeros(Int, ncl), zeros(Int, ncl)
    partial = zeros(Int, ncl * 2 * nchunk)
    # Integer addition, so splitting a bank across chunks must reproduce the serial sum exactly, not
    # approximately. A failure here means the chunk arithmetic drops or double-counts clauses — the
    # kind of bug that would otherwise surface only as slightly wrong training.
    for x in X[1:20]
        TMCore._scores_threaded!(pv, nv, partial, m, x, nchunk)
        for ci in 1:ncl
            p, n = vote(m, ci, x)
            @test pv[ci] == p
            @test nv[ci] == n
            @test pv[ci] - nv[ci] == score(m, ci, x)
        end
    end
end

@testset "the work gate scales with width, not just clause count" begin
    # The bug this guards against: a threshold in clauses alone made the gate fire at the same clause
    # count regardless of how much work a clause is, so a narrow model threaded when it should not.
    narrow = TMClassifier([1, 2], 64; clauses_per_class=400, T=12, S=2, L=10, LF=4)
    wide = TMClassifier([1, 2], 8192; clauses_per_class=400, T=12, S=248, L=10, LF=4)
    @test !TMCore._worth_threading(narrow)
    @test TMCore._worth_threading(wide)
    # And no chunk may be trivially small, which is what turned a missing gate into a 5x slowdown.
    @test TMCore._score_chunks(2, 10) == 1
    @test TMCore._score_chunks(2, 1000) <= 1000 ÷ TMCore.MIN_CLAUSES_PER_CHUNK
end

@testset "shapes and guards" begin
    X, Y = toy(50, 16, 13)
    m = mk(Y, 16)
    @test_throws DimensionMismatch train!(m, X, Y[1:10])
    @test_throws DimensionMismatch train!(m, X, Y[1:10]; parallel=:classes)
    @test_throws ArgumentError train!(m, X, Y; parallel=:examples)   # the racy axis is not offered
    @test_throws ArgumentError train!(m, X, Y; parallel=:nonsense)
    # shuffle=false must still work on both paths; `order` is materialised for the threaded one and
    # an eachindex range would otherwise be shared across tasks.
    train!(m, X, Y; shuffle=false, parallel=:classes)
    train!(m, X, Y; shuffle=false, parallel=:clauses)
    train!(m, X, Y; shuffle=false, parallel=:none)
    @test length(predict(m, X)) == length(X)
end

end
