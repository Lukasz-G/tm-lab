using Test
using TMBoolean

@testset "NGram" begin

DOCS = ["the cat sat on the mat",
        "the dog sat on the log",
        "a cat and a dog",
        "the cat"]

@testset "unigram vocabulary and presence" begin
    e = NGram(n=1, max_features=100)
    fit!(e, DOCS)
    v = feature_names(e)
    @test "the" in v && "cat" in v && "dog" in v
    @test width(e) == length(v)

    B = transform(e, DOCS)
    @test size(B) == (4, width(e))
    @test B isa BitMatrix

    ci = findfirst(==("cat"), v)
    di = findfirst(==("dog"), v)
    @test B[:, ci] == BitVector([true, false, true, true])
    @test B[:, di] == BitVector([false, true, true, false])
end

@testset "presence, not counts" begin
    # "the" appears twice in document 1 and once in document 4; both are a single set bit. A count
    # encoder would differ here, and a TM literal cannot represent a count anyway.
    e = NGram(n=1)
    fit!(e, DOCS)
    B = transform(e, DOCS)
    ti = findfirst(==("the"), feature_names(e))
    @test B[1, ti] && B[4, ti]
end

@testset "n = 2 keeps unigrams as well as bigrams" begin
    e = NGram(n=2, max_features=1000)
    fit!(e, DOCS)
    v = feature_names(e)
    @test "cat" in v
    @test "the cat" in v
    @test "sat on" in v
    # Bigrams do not cross the gaps between documents.
    @test !("mat the" in v)
end

@testset "min_df and max_features" begin
    # "mat" appears in one document only.
    e = NGram(n=1, min_df=2)
    fit!(e, DOCS)
    @test !("mat" in feature_names(e))
    @test "cat" in feature_names(e)

    # The cap keeps the most document-frequent grams. "the" is in 3 of 4 documents.
    e2 = NGram(n=1, max_features=2)
    fit!(e2, DOCS)
    @test width(e2) == 2
    @test "the" in feature_names(e2)
end

@testset "vocabulary is reproducible under ties" begin
    # Every gram here has document frequency 1, so ordering rests entirely on the tie-break. Two
    # encoders fitted on the same corpus must agree, or a model's literal indices mean different
    # things on different runs.
    docs = ["alpha", "beta", "gamma", "delta"]
    a, b = NGram(n=1), NGram(n=1)
    fit!(a, docs); fit!(b, docs)
    @test feature_names(a) == feature_names(b)
    @test issorted(feature_names(a))
end

@testset "unseen words at transform time are dropped" begin
    e = NGram(n=1, max_features=3)
    fit!(e, DOCS)
    B = transform(e, ["completely unseen vocabulary"])
    @test size(B) == (1, width(e))
    @test !any(B)
end

@testset "lowercase" begin
    on = NGram(n=1); fit!(on, ["The THE the"])
    @test feature_names(on) == ["the"]
    off = NGram(n=1, lowercase=false); fit!(off, ["The THE the"])
    @test length(feature_names(off)) == 3
end

@testset "single-document transform" begin
    e = NGram(n=1)
    fit!(e, DOCS)
    @test transform(e, DOCS[2]) == transform(e, DOCS)[2, :]
end

@testset "fit-once discipline and validation" begin
    e = NGram(n=1)
    @test !isfitted(e)
    @test_throws ArgumentError width(e)
    @test_throws ArgumentError feature_names(e)
    @test_throws ArgumentError transform(e, DOCS)

    fit!(e, DOCS)
    @test isfitted(e)
    @test_throws ArgumentError fit!(e, DOCS)
    refit!(e, DOCS)
    @test isfitted(e)

    @test_throws ArgumentError NGram(n=0)
    @test_throws ArgumentError NGram(max_features=0)
    @test_throws ArgumentError NGram(min_df=0)
    @test_throws ArgumentError fit!(NGram(), String[])
    # A min_df no document can satisfy leaves an empty vocabulary, which would produce a zero-width
    # input and a model that cannot learn. Better to say so than to return it.
    @test_throws ArgumentError fit!(NGram(min_df=99), DOCS)
end

end
