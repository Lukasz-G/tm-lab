function roundtrip(m; include_states = true)
    io = IOBuffer()
    save_model(io, m; include_states = include_states)
    return load_model(IOBuffer(take!(io)))
end

@testset "model format round trip" begin
    rng = MersenneTwister(4)
    width = 100                                  # not a multiple of 64: exercises the padded chunk
    V = [rand(rng, Bool, width) for _ in 1:400]
    Y = [v[1] & !v[2] for v in V]
    X = TMInput.(V)

    m = TMClassifier(Y, width; clauses_per_class = 12, T = 8, S = 10, L = 6, LF = 4)
    for _ in 1:8
        train!(m, X, Y; rng = rng)
    end

    back = roundtrip(m)
    @test back.classes == m.classes
    @test back.params.T == m.params.T && back.params.S == m.params.S
    @test back.params.L == m.params.L && back.params.LF == m.params.LF
    @test back.params.width == m.params.width
    @test literal_counts(back) == literal_counts(m)
    @test trainable(back.positive[1])
    for ci in eachindex(m.classes)
        @test back.positive[ci].included == m.positive[ci].included
        @test back.positive[ci].included_inv == m.positive[ci].included_inv
        @test back.negative[ci].state == m.negative[ci].state
        @test back.positive[ci].include_limit == m.positive[ci].include_limit
    end
    @test all(predict(back, x) == predict(m, x) for x in X)

    # A reloaded model must be trainable, not merely readable.
    train!(back, X, Y; rng = MersenneTwister(1))
    @test any(literal_counts(back) .!= literal_counts(m))

    # Inference-only is much smaller and predicts identically, but cannot be trained further.
    lean = roundtrip(m; include_states = false)
    @test !trainable(lean.positive[1])
    @test all(predict(lean, x) == predict(m, x) for x in X)
    @test literal_counts(lean) == literal_counts(m)

    io_full, io_lean = IOBuffer(), IOBuffer()
    save_model(io_full, m); save_model(io_lean, m; include_states = false)
    @test io_lean.size * 4 < io_full.size      # automata dominate: one byte per position per clause
    @test_throws ArgumentError save_model(IOBuffer(), lean)   # no automata to save
end

@testset "policy types survive the format" begin
    rng = MersenneTwister(5)
    V = [rand(rng, Bool, 64) for _ in 1:200]
    Y = [v[1] for v in V]
    X = TMInput.(V)
    for ceil in (LiteralCapped(), FlatLF()), bud in (GrowthGate(), HardCap())
        m = TMClassifier(Y, 64; clauses_per_class = 8, T = 6, S = 8, L = 5, LF = 3,
                         ceiling = ceil, budget = bud)
        train!(m, X, Y; rng = rng)
        back = roundtrip(m)
        # The policies are the whole reason this type exists; losing them on save would silently
        # change the model's semantics on reload.
        @test typeof(back.ceiling) == typeof(ceil)
        @test typeof(back.budget) == typeof(bud)
        @test all(predict(back, x) == predict(m, x) for x in X)
    end
end

@testset "class label types" begin
    rng = MersenneTwister(6)
    V = [rand(rng, Bool, 64) for _ in 1:200]
    X = TMInput.(V)
    for Y in (Bool[v[1] for v in V],
              Int[v[1] ? 7 : -3 for v in V],
              String[v[1] ? "yes" : "no" for v in V])
        m = TMClassifier(Y, 64; clauses_per_class = 6, T = 6, S = 8, L = 5, LF = 3)
        train!(m, X, Y; rng = rng)
        back = roundtrip(m)
        @test back.classes == m.classes
        @test eltype(back.classes) == eltype(m.classes)
        @test all(predict(back, x) == predict(m, x) for x in X)
    end
end

@testset "format rejects what it cannot trust" begin
    m = TMClassifier([false, true], 64; clauses_per_class = 4, T = 4, S = 8, L = 4, LF = 2)
    io = IOBuffer(); save_model(io, m)
    good = take!(io)

    @test_throws ArgumentError load_model(IOBuffer(b"not a model at all, really"))

    bad_version = copy(good)
    bad_version[9] = 0x63                       # version 99
    err = try load_model(IOBuffer(bad_version)) catch e; e end
    @test err isa ArgumentError && occursin("unsupported", err.msg)

    # Truncation must fail loudly rather than yield a plausible-looking half model.
    @test_throws Exception load_model(IOBuffer(good[1:end - 64]))

    # A header that disagrees with itself is rejected before any array is read.
    bad_dims = copy(good)
    bad_dims[21] = 0xff                          # nchunks, byte 20 zero-based
    @test_throws ArgumentError load_model(IOBuffer(bad_dims))
end
