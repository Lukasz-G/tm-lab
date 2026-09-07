using Test
using TMBoolean
using Random: MersenneTwister

@testset "TMBoolean" begin

@testset "thermometer shape and monotonicity" begin
    rng = MersenneTwister(1)
    X = rand(rng, 200, 5)
    t = Thermometer(nthresholds = 4)
    @test !isfitted(t)
    @test_throws ArgumentError width(t)
    @test_throws ArgumentError feature_names(t)
    @test_throws ArgumentError transform(t, X)

    fit!(t, X)
    @test isfitted(t)
    @test width(t) == 5 * 4
    @test length(feature_names(t)) == 20

    B = transform(t, X)
    @test size(B) == (200, 20)
    @test B isa BitMatrix

    # Thresholds ascend within a feature, so the bits are nested: x > t3 implies x > t2. This is the
    # property that lets a conjunctive clause express an inequality with one literal and an interval
    # with two; one-hot bins would need a disjunction, which a clause cannot represent.
    for j in 1:5
        @test issorted(t.thresholds[j, :])
        for i in 1:200, c in 2:4
            B[i, (j - 1) * 4 + c] && @test B[i, (j - 1) * 4 + c - 1]
        end
    end
end

@testset "quantile thresholds split the data evenly" begin
    rng = MersenneTwister(2)
    X = reshape(sort(rand(rng, 1000)), :, 1)
    t = Thermometer(nthresholds = 3, strategy = :quantile)
    B = fit_transform!(t, X)
    # Interior quartiles: bits should be on for about 75%, 50% and 25% of samples.
    for (c, want) in zip(1:3, (0.75, 0.50, 0.25))
        @test isapprox(count(B[:, c]) / 1000, want; atol = 0.03)
    end
end

@testset "uniform thresholds follow the scale, not the mass" begin
    # Heavily skewed: 990 values near 0, 10 near 1. Quantile puts all its thresholds in the dense
    # region; uniform spreads them across the range and leaves most bits nearly dead. Both are
    # defensible and they are not interchangeable, which is why the strategy is explicit.
    x = vcat(fill(0.01, 990), fill(1.0, 10))
    X = reshape(x, :, 1)
    q = fit_transform!(Thermometer(nthresholds = 3, strategy = :quantile), X)
    u = fit_transform!(Thermometer(nthresholds = 3, strategy = :uniform), X)
    @test count(u[:, 1]) == 10                      # every uniform bit only catches the tail
    @test count(q[:, 1]) >= count(u[:, 1])
end

@testset "constant features produce no information, and do not error" begin
    X = fill(3.0, 50, 2)
    t = Thermometer(nthresholds = 4)
    B = fit_transform!(t, X)
    @test size(B) == (50, 8)
    @test !any(B)                                   # x > x is false everywhere
end

@testset "fit-once is enforced, refit is deliberate" begin
    rng = MersenneTwister(3)
    Xtr, Xte = rand(rng, 100, 3), rand(rng, 40, 3)
    t = Thermometer(nthresholds = 2)
    fit!(t, Xtr)

    # The guard exists because refitting on later data is silent lookahead leakage: no error, no
    # warning, just a better-looking score. Requiring a different function name is the cheap fix.
    err = try fit!(t, Xte) catch e; e end
    @test err isa ArgumentError
    @test occursin("lookahead", err.msg)
    @test occursin("refit!", err.msg)

    before = copy(t.thresholds)
    refit!(t, Xte)
    @test t.thresholds != before
    @test isfitted(t)
end

@testset "transform is fixed by the fit, not by the data it sees" begin
    # The whole point of fit/transform separation: test data must be encoded with train thresholds.
    rng = MersenneTwister(4)
    Xtr = rand(rng, 500, 2)
    Xte = rand(rng, 500, 2) .+ 10.0                 # entirely outside the training range
    t = Thermometer(nthresholds = 3)
    fit!(t, Xtr)
    B = transform(t, Xte)
    @test all(B)                                    # everything is above every train threshold
    @test_throws DimensionMismatch transform(t, rand(rng, 5, 7))
end

@testset "single-sample transform matches the matrix path" begin
    rng = MersenneTwister(5)
    X = rand(rng, 50, 4)
    t = Thermometer(nthresholds = 3)
    B = fit_transform!(t, X)
    for i in (1, 17, 50)
        @test transform(t, X[i, :]) == B[i, :]
    end
end

@testset "feature names identify the bit" begin
    t = Thermometer(nthresholds = 2)
    fit!(t, reshape(Float64[0, 1, 2, 3], :, 1))
    n = feature_names(t)
    @test length(n) == 2
    @test all(startswith.(n, "f1>"))
    # Names must be in bit order, so name[i] describes column i and a literal index reads back.
    @test n[1] != n[2]
end

@testset "constructor validation" begin
    @test_throws ArgumentError Thermometer(nthresholds = 0)
    @test_throws ArgumentError Thermometer(strategy = :nonsense)
end

end
