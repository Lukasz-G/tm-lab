# CIFAR-10 loader, binary distribution.
#
# The point of adding it is that MNIST and Fashion-MNIST are 28x28 grayscale in the same file format
# with the same class count — nearer two samples of one problem than two problems. CIFAR-10 is
# 32x32 colour with real backgrounds, so a conclusion that survives it has actually travelled.
#
# Format (cs.toronto.edu/~kriz/cifar.html): each record is 1 label byte followed by 3072 pixel
# bytes, ordered 1024 red, then green, then blue, row-major within a channel. 10,000 records per
# file, five training files and one test file.

const _CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
const _CIFAR_DIR = "cifar-10-batches-bin"

function _cifar_ensure(datadir)
    root = joinpath(datadir, _CIFAR_DIR)
    isdir(root) && isfile(joinpath(root, "test_batch.bin")) && return root
    mkpath(datadir)
    tgz = joinpath(datadir, "cifar-10-binary.tar.gz")
    isfile(tgz) || (@info "downloading CIFAR-10 (~170 MB)"; download(_CIFAR_URL, tgz))
    # `tar` ships with Windows 10+ and every unix; avoids pulling in a compression package for a
    # one-off extraction.
    run(Cmd(`tar -xzf $tgz -C $datadir`))
    return root
end

"""
    cifar10(datadir, split) -> (X, y)

`split` is `:train` (50,000) or `:test` (10,000). Returns `X` as an `nsamples x 3072` `Matrix{UInt8}`
in the file's native R,G,B channel order and `y` as `Vector{Int}` in `0:9`.

Pixels are returned raw rather than booleanized: how to threshold colour is a real choice and belongs
to the caller, not to a loader.
"""
function cifar10(datadir, split::Symbol)
    root = _cifar_ensure(datadir)
    files = split === :train ? ["data_batch_$(i).bin" for i in 1:5] :
            split === :test ? ["test_batch.bin"] :
            throw(ArgumentError("split must be :train or :test, got :$split"))
    n = 10_000 * length(files)
    X = Matrix{UInt8}(undef, n, 3072)
    y = Vector{Int}(undef, n)
    row = 0
    for f in files
        bytes = read(joinpath(root, f))
        length(bytes) == 10_000 * 3073 || error("unexpected size for $f")
        @inbounds for k in 0:9_999
            off = k * 3073
            row += 1
            y[row] = Int(bytes[off + 1])
            copyto!(view(X, row, :), view(bytes, (off + 2):(off + 3073)))
        end
    end
    return X, y
end

"""
    cifar_gray(X) -> Matrix{Float64}

Luminance, `nsamples x 1024`, in `[0,1]`. Cheap way to cut the input width by three when colour is
not what is being tested — a variant comparison does not need colour, only a baseline that learns.
"""
function cifar_gray(X::AbstractMatrix{UInt8})
    n = size(X, 1)
    G = Matrix{Float64}(undef, n, 1024)
    @inbounds for i in 1:n, p in 1:1024
        r, g, b = X[i, p], X[i, 1024 + p], X[i, 2048 + p]
        G[i, p] = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    end
    return G
end

"`nsamples x 3072` scaled to [0,1], colour retained."
cifar_rgb(X::AbstractMatrix{UInt8}) = Float64.(X) ./ 255
