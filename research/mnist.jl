# Shared MNIST loader for research/ scripts.
#
# Fetched straight from the idx files rather than through MLDatasets, so experiments carry no heavy
# data dependency. Files land in gitignored data/.

const _MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"

function _fetch_idx(datadir, name)
    mkpath(datadir)
    gz, raw = joinpath(datadir, name * ".gz"), joinpath(datadir, name)
    if !isfile(raw)
        isfile(gz) || download("$_MNIST_MIRROR/$name.gz", gz)
        open(raw, "w") do out
            write(out, read(pipeline(`gzip -dc $gz`)))
        end
    end
    return read(raw)
end

_be32(b, i) = (Int(b[i]) << 24) | (Int(b[i+1]) << 16) | (Int(b[i+2]) << 8) | Int(b[i+3])

"""
    mnist_test(datadir) -> (images, labels, n, nrows, ncols)

Raw MNIST test set. `images` is a `ncols × nrows × n` UInt8 array — note that reshaping the
row-major idx buffer in Julia's column-major order lands transposed relative to the file, which is
in fact the layout the upstream models were trained on. Callers should still confirm orientation by
accuracy rather than trusting this comment; see [`booleanize_both`](@ref).
"""
function mnist_test(datadir)
    ib = _fetch_idx(datadir, "t10k-images-idx3-ubyte")
    lb = _fetch_idx(datadir, "t10k-labels-idx1-ubyte")
    _be32(ib, 1) == 2051 || error("bad image magic")
    _be32(lb, 1) == 2049 || error("bad label magic")
    n, nr, nc = _be32(ib, 5), _be32(ib, 9), _be32(ib, 13)
    px = reshape(ib[17:16+n*nr*nc], nc, nr, n)
    return px, Int8.(lb[9:8+n]), n, nr, nc
end

"""
    mnist_train(datadir) -> (images, labels, n, nrows, ncols)

Training split, same layout as [`mnist_test`](@ref).
"""
function mnist_train(datadir)
    ib = _fetch_idx(datadir, "train-images-idx3-ubyte")
    lb = _fetch_idx(datadir, "train-labels-idx1-ubyte")
    _be32(ib, 1) == 2051 || error("bad image magic")
    _be32(lb, 1) == 2049 || error("bad label magic")
    n, nr, nc = _be32(ib, 5), _be32(ib, 9), _be32(ib, 13)
    px = reshape(ib[17:16+n*nr*nc], nc, nr, n)
    return px, Int8.(lb[9:8+n]), n, nr, nc
end

"""
    booleanize_both(px, n) -> (raw, transposed)

Both candidate orientations, booleanized at the upstream threshold (`x > 0.25` on Float32 in
`[0,1]`, i.e. byte > 63). A wrong transpose permutes every bit and fails silently, so callers pick
between these by measured accuracy rather than by assumption.
"""
function booleanize_both(px, n)
    raw   = [BitVector(vec(view(px, :, :, i)) .> 0x3f) for i in 1:n]
    trans = [BitVector(vec(permutedims(view(px, :, :, i))) .> 0x3f) for i in 1:n]
    return raw, trans
end
