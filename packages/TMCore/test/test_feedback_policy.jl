@testset "feedback policy branch selection" begin
    rng = MersenneTwister(31)

    # The published rule ignores magnitude entirely: any nonzero vote reinforces.
    for v in 1:5
        @test reinforce_branch(ThresholdFeedback(), v, 5, rng)
        @test reject_branch(ThresholdFeedback(), v, 5, rng)
    end
    @test !reinforce_branch(ThresholdFeedback(), 0, 5, rng)
    @test !reject_branch(ThresholdFeedback(), 0, 5, rng)

    # Proportional agrees at both ends — a zero vote never reinforces, a full match always does —
    # so it is a strict generalisation rather than a different rule.
    for pol in (ProportionalFeedback(),)
        @test !reinforce_branch(pol, 0, 5, rng)
        @test !reject_branch(pol, 0, 5, rng)
        for _ in 1:200
            @test reinforce_branch(pol, 5, 5, rng)
            @test reinforce_branch(pol, 7, 5, rng)      # vote above ceiling cannot happen, but is safe
            @test reject_branch(pol, 5, 5, rng)
        end
    end

    # In between it fires at the stated rate. 4000 draws puts the standard error near 0.008.
    for (v, ceil) in ((1, 5), (2, 5), (3, 5), (4, 5), (1, 2))
        hits = count(_ -> reinforce_branch(ProportionalFeedback(), v, ceil, rng), 1:4000)
        @test isapprox(hits / 4000, v / ceil; atol = 0.035)
    end
end

@testset "proportional feedback trains" begin
    rng = MersenneTwister(32)
    width, n = 24, 900
    V = [rand(rng, Bool, width) for _ in 1:n]
    Y = [v[1] & !v[3] for v in V]
    X = TMInput.(V)

    for pol in (ThresholdFeedback(), ProportionalFeedback())
        m = TMClassifier(Y, width; clauses_per_class = 16, T = 8, S = 10, L = 6, LF = 4,
                         feedback = pol)
        r = MersenneTwister(5)
        for _ in 1:12
            train!(m, X, Y; rng = r)
        end
        # Both must actually learn. Baseline here is 0.75, so anything near it is not learning.
        @test accuracy(predict(m, X), Y) > 0.95
        @test any(>(0), literal_counts(m))
    end
end

@testset "feedback policy survives the format" begin
    rng = MersenneTwister(33)
    V = [rand(rng, Bool, 64) for _ in 1:200]
    Y = [v[1] for v in V]
    X = TMInput.(V)
    for pol in (ThresholdFeedback(), ProportionalFeedback())
        m = TMClassifier(Y, 64; clauses_per_class = 8, T = 6, S = 8, L = 5, LF = 3, feedback = pol)
        train!(m, X, Y; rng = rng)
        io = IOBuffer(); save_model(io, m)
        back = load_model(IOBuffer(take!(io)))
        # Inference does not depend on the feedback policy, but continuing training does — losing it
        # on reload would silently switch rules mid-run.
        @test typeof(back.feedback) == typeof(pol)
        @test all(predict(back, x) == predict(m, x) for x in X)
    end
end
