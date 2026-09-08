using Test
using TMBoolean
using Random: MersenneTwister

@testset "ConvKernel" begin

# A hand-computed convolution, because everything else here is shape arithmetic that would pass just
# as happily against an encoder that convolves the wrong way round or transposes the image.
@testset "valid convolution against a hand computation" begin
    # 4x4 image, row-major, with a vertical step from 0 to 1 between columns 2 and 3.
    img = Float64[0 0 1 1
                  0 0 1 1
                  0 0 1 1
                  0 0 1 1]
    X = reshape(permutedims(img)[:], 1, 16)   # row-major flatten, one sample

    e = ConvKernel(height=4, width=4, kernels=[TMBoolean._SOBEL_X], nthresholds=1, rectify=false)
    R = responses(e, X)
    # Valid 3x3 on 4x4 gives 2x2 responses. Sobel-x over a left-to-right 0->1 step: the window
    # centred at column 2 spans columns 1..3 = [0 0 1], giving -1*0 -2*0 -1*0 + 1*1 +2*1 +1*1 = 4.
    # The window centred at column 3 spans [0 1 1], giving 1+2+1 - 0 = 4 as well... except the left
    # column is 0 and the right is 1 in both, so both responses are 4.
    @test size(R) == (1, 4)
    @test all(R .== 4.0)

    # Sobel-y on a purely vertical edge must be flat zero. This is the assertion that catches a
    # transposed image, which the x-kernel alone cannot.
    ey = ConvKernel(height=4, width=4, kernels=[TMBoolean._SOBEL_Y], nthresholds=1, rectify=false)
    @test all(responses(ey, X) .== 0.0)
end

@testset "rectify" begin
    img = Float64[1 1 0 0; 1 1 0 0; 1 1 0 0; 1 1 0 0]      # step the other way: response -4
    X = reshape(permutedims(img)[:], 1, 16)
    signed = ConvKernel(height=4, width=4, kernels=[TMBoolean._SOBEL_X], rectify=false)
    rect = ConvKernel(height=4, width=4, kernels=[TMBoolean._SOBEL_X], rectify=true)
    @test all(responses(signed, X) .== -4.0)
    @test all(responses(rect, X) .== 4.0)
end

@testset "shape arithmetic, pooling and channels" begin
    # 3x3 kernels, valid, on 8x8 -> 6x6 per kernel.
    e = ConvKernel(height=8, width=8, channels=1, kernels=:edges, nthresholds=2, pool=1)
    @test nresponses(e) == 3 * 36
    X = rand(MersenneTwister(2), 20, 64)
    fit!(e, X)
    @test width(e) == 3 * 36 * 2
    @test length(feature_names(e)) == width(e)
    @test size(transform(e, X)) == (20, width(e))

    # pool 2 on a 6x6 response map -> 3x3.
    e2 = ConvKernel(height=8, width=8, kernels=:sobel, nthresholds=1, pool=2)
    @test nresponses(e2) == 2 * 9

    # pool 4 does not divide 6: the last window is partial and must still be pooled, not dropped.
    e3 = ConvKernel(height=8, width=8, kernels=:sobel, nthresholds=1, pool=4)
    @test nresponses(e3) == 2 * 4          # cld(6,4) == 2 per side

    # Channels multiply the response count and nothing else.
    e4 = ConvKernel(height=8, width=8, channels=3, kernels=:sobel, nthresholds=1, pool=2)
    @test nresponses(e4) == 3 * 2 * 9
end

@testset "channels are read channel-major" begin
    # Channel 1 is a vertical edge, channel 2 is flat. Sobel-x must fire on the first and not the
    # second, which pins the channel layout rather than assuming it.
    edge = Float64[0 0 1 1; 0 0 1 1; 0 0 1 1; 0 0 1 1]
    flat = zeros(4, 4)
    X = reshape(vcat(permutedims(edge)[:], permutedims(flat)[:]), 1, 32)
    e = ConvKernel(height=4, width=4, channels=2, kernels=[TMBoolean._SOBEL_X], nthresholds=1)
    R = responses(e, X)
    @test size(R) == (1, 8)
    @test all(R[1, 1:4] .== 4.0)
    @test all(R[1, 5:8] .== 0.0)
end

@testset "constant image gives zero response" begin
    # Every kernel in the bank sums to zero, so a flat image must produce exactly zero everywhere.
    # If a kernel is ever added that does not sum to zero this test should be updated, not deleted.
    X = fill(0.7, 3, 25)
    e = ConvKernel(height=5, width=5, kernels=:edges, nthresholds=1)
    @test all(abs.(responses(e, X)) .< 1e-12)
end

@testset "fit-once discipline" begin
    rng = MersenneTwister(3)
    X = rand(rng, 30, 36)
    e = ConvKernel(height=6, width=6, kernels=:sobel, nthresholds=2)
    @test !isfitted(e)
    @test_throws ArgumentError width(e)
    @test_throws ArgumentError feature_names(e)
    @test_throws ArgumentError transform(e, X)

    fit!(e, X)
    @test isfitted(e)
    @test_throws ArgumentError fit!(e, X)      # refitting must be deliberate

    w = width(e)
    refit!(e, rand(rng, 30, 36))
    @test isfitted(e)
    @test width(e) == w
end

@testset "argument validation" begin
    @test_throws ArgumentError ConvKernel(height=0, width=4)
    @test_throws ArgumentError ConvKernel(height=4, width=4, channels=0)
    @test_throws ArgumentError ConvKernel(height=4, width=4, pool=0)
    @test_throws ArgumentError ConvKernel(height=4, width=4, kernels=:nonsense)
    @test_throws ArgumentError ConvKernel(height=4, width=4, kernels=Matrix{Float64}[])
    # Even-sided kernels have no centre, so the valid-convolution indexing is ambiguous.
    @test_throws ArgumentError ConvKernel(height=4, width=4, kernels=[ones(2, 2)])
    # A kernel larger than the image.
    @test_throws ArgumentError ConvKernel(height=2, width=2, kernels=:edges)

    e = ConvKernel(height=4, width=4, kernels=:sobel, nthresholds=1)
    @test_throws DimensionMismatch fit!(e, rand(5, 15))
end

@testset "single-sample transform" begin
    rng = MersenneTwister(4)
    X = rand(rng, 40, 36)
    e = ConvKernel(height=6, width=6, kernels=:sobel, nthresholds=2, pool=2)
    fit!(e, X)
    B = transform(e, X)
    @test transform(e, X[1, :]) == B[1, :]
end

@testset "names identify kernel, channel and position" begin
    e = ConvKernel(height=6, width=6, channels=2, kernels=:sobel, nthresholds=1, pool=2)
    fit!(e, rand(MersenneTwister(5), 10, 72))
    n = feature_names(e)
    @test length(n) == nresponses(e)
    @test startswith(n[1], "c1.sobelx[1,1]>")
    @test any(s -> startswith(s, "c2.sobely["), n)
end

end
