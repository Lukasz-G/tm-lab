@testset "observe reproduces the evaluator exactly" begin
    rng = MersenneTwister(8)
    width = 70                                   # spans two chunks, second one padded
    V = [rand(rng, Bool, width) for _ in 1:300]
    Y = [v[1] & !v[3] for v in V]
    X = TMInput.(V)

    m = TMClassifier(Y, width; clauses_per_class = 10, T = 6, S = 8, L = 6, LF = 4)
    for _ in 1:6
        train!(m, X, Y; rng = rng)
    end

    st = observe(m, X)
    @test st.nexamples == length(X)
    @test size(st.votes, 1) == m.params.LF + 1
    @test size(st.satisfied, 1) == width

    # Every recorded vote must equal what the evaluator returns, and every satisfied count must
    # equal a naive recomputation. Recording is only useful if it is faithful.
    for (k, (ci, pol, j)) in enumerate(st.slot_of)
        b = pol == 1 ? m.positive[ci] : m.negative[ci]
        @test st.counts[k] == Int(b.count[j])
        @test st.ceilings[k] == ceiling(m.ceiling, st.counts[k], m.params.LF)
        @test sum(@view st.votes[:, k]) == length(X)

        hist = zeros(Int, m.params.LF + 1)
        sat = zeros(Int, width)
        for (xi, x) in enumerate(X)
            v = clause_vote(b, j, x, m.params.LF, m.ceiling)
            hist[v + 1] += 1
            for i in 1:width
                n, bit = (i - 1) >> 6 + 1, (i - 1) & 63
                inc = !iszero(b.included[n, j] & (one(UInt64) << bit))
                inv = !iszero(b.included_inv[n, j] & (one(UInt64) << bit))
                if (inc && !inv && V[xi][i]) || (inv && !inc && !V[xi][i])
                    sat[i] += 1
                end
            end
        end
        @test st.votes[:, k] == hist
        @test st.satisfied[:, k] == sat
    end

    # A literal the clause does not include can never be recorded as satisfied.
    for (k, (ci, pol, j)) in enumerate(st.slot_of)
        b = pol == 1 ? m.positive[ci] : m.negative[ci]
        for i in 1:width
            n, bit = (i - 1) >> 6 + 1, (i - 1) & 63
            included = !iszero((b.included[n, j] | b.included_inv[n, j]) & (one(UInt64) << bit))
            included || @test st.satisfied[i, k] == 0
        end
    end

    @test sum(vote_histogram(st)) == length(X) * length(st.slot_of)
    @test vote_histogram(st) == [sum(@view st.votes[v + 1, :]) for v in 0:m.params.LF]
end

@testset "interior fraction and spread behave at the extremes" begin
    width = 64
    # A strict model, LF = 1: a clause can only vote 0 or its ceiling, so nothing is ever interior.
    rng = MersenneTwister(9)
    V = [rand(rng, Bool, width) for _ in 1:200]
    Y = [v[1] for v in V]
    X = TMInput.(V)

    strict = TMClassifier(Y, width; clauses_per_class = 8, T = 6, S = 8, L = 5, LF = 1)
    for _ in 1:5
        train!(strict, X, Y; rng = rng)
    end
    @test all(iszero, interior_fraction(observe(strict, X)))

    # A fuzzy model on the same data should put some mass in the interior.
    fuzzy = TMClassifier(Y, width; clauses_per_class = 8, T = 6, S = 8, L = 5, LF = 4)
    for _ in 1:5
        train!(fuzzy, X, Y; rng = rng)
    end
    st = observe(fuzzy, X)
    @test any(>(0), interior_fraction(st))
    @test all(0 .<= interior_fraction(st) .<= 1)
    @test all(0 .<= satisfaction_spread(st) .<= 1)

    # An untrained model has no included literals, so every clause votes its ceiling every time:
    # no interior, no spread, and the histogram is a single spike.
    empty = TMClassifier(Y, width; clauses_per_class = 8, T = 6, S = 8, L = 5, LF = 4)
    ste = observe(empty, X)
    @test all(iszero, interior_fraction(ste))
    @test all(iszero, satisfaction_spread(ste))
    h = vote_histogram(ste)
    @test h[end] == length(X) * length(ste.slot_of) && all(iszero, h[1:end-1])
end

@testset "benchmark harness" begin
    rng = MersenneTwister(10)
    V = [rand(rng, Bool, 128) for _ in 1:400]
    Y = [v[1] for v in V]
    X = TMInput.(V)
    m = TMClassifier(Y, 128; clauses_per_class = 10, T = 6, S = 8, L = 5, LF = 3)
    train!(m, X, Y; rng = rng)

    r = benchmark(m, X, Y; loops = 3)
    @test r.nexamples == length(X)
    @test r.predict_seconds > 0
    @test r.predictions_per_second > 0
    @test r.accuracy == accuracy(predict(m, X), Y)
    @test r.train_epoch_seconds === nothing
    @test occursin("predictions/s", sprint(show, r))

    r2 = benchmark(m, X, Y; loops = 2, train = true, rng = rng)
    @test r2.train_epoch_seconds !== nothing && r2.train_epoch_seconds > 0
    @test_throws ArgumentError benchmark(m, X; train = true)

    sz = model_bytes(m)
    @test sz.with_states > sz.inference_only
    # Automata are a byte per position per clause; masks are a bit. The ratio should be large.
    @test sz.with_states > 4 * sz.inference_only
end
