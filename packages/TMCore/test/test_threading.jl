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

@testset "clause parallelism is deterministic, and works where classes cannot" begin
    # Two classes and a wide bank: `:classes` has almost nothing to spread here, which is the whole
    # reason `:clauses` exists. Above PARALLEL_MIN_CLAUSES so the threaded branch is actually taken
    # rather than silently falling back to the serial one.
    rng = MersenneTwister(21)
    X = [TMInput(rand(rng, Bool, 32)) for _ in 1:400]
    Y = [x[1] ⊻ x[2] ? 1 : 2 for x in X]          # binary, and not linearly separable
    mk2() = TMClassifier(Y, 32; clauses_per_class=200, T=12, S=20, L=10, LF=4)
    @test 100 >= TMCore.PARALLEL_MIN_CLAUSES      # 200 per class splits into 100 per polarity

    a, b = mk2(), mk2()
    for _ in 1:3
        train!(a, X, Y; rng=MersenneTwister(4), parallel=:clauses)
        train!(b, X, Y; rng=MersenneTwister(4), parallel=:clauses)
    end
    @test fingerprint(a) == fingerprint(b)

    # Distinct from both other schedules, for the documented RNG-stream reason.
    c = mk2()
    train!(c, X, Y; rng=MersenneTwister(4), parallel=:none)
    d = mk2()
    train!(d, X, Y; rng=MersenneTwister(4), parallel=:classes)
    e = mk2()
    train!(e, X, Y; rng=MersenneTwister(4), parallel=:clauses)
    @test fingerprint(c) != fingerprint(e)
    @test fingerprint(d) != fingerprint(e)

    # And it must still learn: a race would usually still satisfy the inequalities above.
    for m in (c, e)
        for _ in 1:10
            train!(m, X, Y; rng=MersenneTwister(2), parallel=(m === e ? :clauses : :none))
        end
        @test accuracy(predict(m, X), Y) > 0.9
    end
end

@testset "threaded bank_vote sums exactly" begin
    rng = MersenneTwister(31)
    X = [TMInput(rand(rng, Bool, 32)) for _ in 1:120]
    Y = [x[1] ? 1 : 2 for x in X]
    m = TMClassifier(Y, 32; clauses_per_class=200, T=12, S=20, L=10, LF=4)
    train!(m, X, Y; rng=MersenneTwister(1))
    # Integer addition, so chunking across threads must reproduce the serial sum exactly — not
    # approximately. If this ever fails the chunk arithmetic is dropping or double-counting clauses.
    for x in X[1:20]
        @test TMCore._bank_vote_threaded(m.positive[1], x, m.params.LF, m.ceiling, m.misscost) ==
              bank_vote(m.positive[1], x, m.params.LF, m.ceiling, m.misscost)
    end
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
