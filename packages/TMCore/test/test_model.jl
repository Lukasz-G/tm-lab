@testset "classifier construction and untrained symmetry" begin
    m = TMClassifier([:a, :b, :c], 64; clauses_per_class = 8, T = 10, S = 20, L = 8, LF = 4)
    @test TMCore.nclasses(m) == 3
    @test length(m.positive) == 3 && m.positive[1].nclauses == 4
    @test m.ceiling isa LiteralCapped && m.budget isa GrowthGate && m.binding isa OneVsRest

    # Untrained, every clause is empty, so every clause votes its ceiling and every class ties at
    # zero. That symmetry is what makes the first update unbiased rather than arbitrary.
    x = TMInput(rand(Bool, 64))
    for ci in 1:3
        p, n = vote(m, ci, x)
        @test p == n == 4 * 4                    # 4 clauses per polarity, each ceilinged at LF = 4
        @test score(m, ci, x) == 0
    end
    @test predict(m, x) == :a                    # ties resolve to the first class

    @test_throws ArgumentError TMClassifier([:a], 64; clauses_per_class = 8, T = 1, S = 1, L = 1, LF = 1)
    @test_throws ArgumentError TMClassifier([:a, :b], 64; clauses_per_class = 1, T = 1, S = 1, L = 1, LF = 1)
    @test_throws ArgumentError train!(m, x, :zzz)
end

@testset "learns a concept" begin
    # y = x1 AND x2 over 20 features, the rest noise.
    rng = MersenneTwister(42)
    width, n = 20, 1200
    V = [rand(rng, Bool, width) for _ in 1:n]
    Y = [v[1] & v[2] for v in V]
    X = TMInput.(V)
    ntr = 900

    m = TMClassifier(Y[1:ntr], width; clauses_per_class = 20, T = 8, S = 10, L = 8, LF = 4)
    for _ in 1:15
        train!(m, X[1:ntr], Y[1:ntr]; rng = rng)
    end
    @test accuracy(predict(m, X[ntr+1:end]), Y[ntr+1:end]) > 0.95

    # Clauses should be real but not degenerate: something included, nothing near-total.
    counts = literal_counts(m)
    @test any(>(0), counts)
    @test maximum(counts) < width * 2

    # LF = 1 is the classical strict TM, and it should handle this too.
    m2 = TMClassifier(Y[1:ntr], width; clauses_per_class = 20, T = 8, S = 10, L = 8, LF = 1)
    for _ in 1:15
        train!(m2, X[1:ntr], Y[1:ntr]; rng = rng)
    end
    @test accuracy(predict(m2, X[ntr+1:end]), Y[ntr+1:end]) > 0.9
end

@testset "ceiling and budget policies both reach training" begin
    rng = MersenneTwister(11)
    width, n = 24, 800
    V = [rand(rng, Bool, width) for _ in 1:n]
    Y = [v[1] & !v[3] for v in V]
    X = TMInput.(V)

    function run(; ceiling, budget, seed = 5)
        m = TMClassifier(Y, width; clauses_per_class = 16, T = 8, S = 10, L = 6, LF = 4,
                         ceiling = ceiling, budget = budget)
        r = MersenneTwister(seed)
        for _ in 1:10
            train!(m, X, Y; rng = r)
        end
        return m
    end

    capped = run(ceiling = LiteralCapped(), budget = GrowthGate())
    flat   = run(ceiling = FlatLF(), budget = GrowthGate())
    hard   = run(ceiling = LiteralCapped(), budget = HardCap())

    # Both growth-gate variants learn this easily. The concept is true 25% of the time, so the
    # majority-class baseline is 0.75 and anything near it is not learning.
    @test accuracy(predict(capped, X), Y) > 0.95
    @test accuracy(predict(flat, X), Y) > 0.95

    # HardCap is pinned deliberately, because it is a measured result rather than a threshold that
    # happened to pass: enforcing L as the documented cap lands around 0.79 against a 0.75 baseline,
    # roughly 20 points behind the growth-gate behaviour the references actually implement. If a
    # later change makes HardCap competitive, this assertion should fail and be investigated, not
    # relaxed.
    hard_acc = accuracy(predict(hard, X), Y)
    @test 0.75 < hard_acc < 0.90

    # The cap has to actually bind, on every path that can grow a clause.
    @test maximum(literal_counts(hard)) <= 6
    @test maximum(literal_counts(capped)) > 6      # growth gate overshoots L, as it does upstream
end

@testset "training is deterministic given a seed" begin
    rng = MersenneTwister(1)
    width, n = 16, 300
    V = [rand(rng, Bool, width) for _ in 1:n]
    Y = [v[1] != v[2] for v in V]
    X = TMInput.(V)
    function run(seed)
        m = TMClassifier(Y, width; clauses_per_class = 10, T = 6, S = 8, L = 6, LF = 3)
        r = MersenneTwister(seed)
        for _ in 1:5
            train!(m, X, Y; rng = r)
        end
        return literal_counts(m)
    end
    @test run(99) == run(99)
    @test run(99) != run(100)
end
