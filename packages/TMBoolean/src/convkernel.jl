# Fixed convolutional kernels applied at booleanization, then thresholded.
#
# This is the encoder the design notes single out: FPTM's Fashion-MNIST state-of-the-art leans on
# fixed convolutional kernels applied before the machine sees anything, and the paper explicitly
# disclaims them as not part of the TM. So the accuracy they carry belongs to the encoder, and the
# only way to say how much that is is to have them as a named, measurable encoder rather than as
# preprocessing buried in a benchmark script.
#
# Nothing here is learned. The kernels are fixed edge and centre-surround operators; the only fitted
# quantities are the thermometer thresholds over their responses.

"""
    ConvKernel(; height, width, channels = 1, kernels = :edges, nthresholds = 2,
                 strategy = :quantile, pool = 1, rectify = true)

Convolve each channel of an image with a bank of fixed kernels, optionally pool, then thermometer-
encode the responses.

Input is `nsamples x (channels * height * width)`, channel-major: channel `c` occupies columns
`(c-1)*height*width .+ (1:height*width)`, and within a channel pixels are row-major. That is CIFAR's
own layout, so no transposition is needed at the call site.

`kernels`:
- `:edges` — Sobel x, Sobel y, and a 4-neighbour Laplacian. Gradients in two directions plus a
  centre-surround operator, which is the smallest bank that is not obviously missing something.
- `:sobel` — the two gradients only.
- a `Vector{Matrix{Float64}}` of odd-sized kernels, for anything else.

Convolution is `:valid`, so a `k x k` kernel on an `h x w` image gives `(h-k+1) x (w-k+1)` responses.

`pool` is a max-pooling factor applied to the response map before encoding. It exists because width
is the binding constraint: on 32x32x3 with three kernels, `pool = 1` produces 2,700 responses per
image and `nthresholds = 2` turns that into 5,400 bits, which is wider than the raw image. Pooling
is how this encoder is made affordable, and it is lossy — say which pooling was used when quoting a
result.

`rectify = true` encodes `abs(response)`, i.e. edge *strength* regardless of direction, and is the
default because max-pooling signed responses lets opposite edges inside one window cancel. Set it to
`false` to keep the sign, which is meaningful for the Laplacian and for a `pool` of 1.
"""
mutable struct ConvKernel <: Booleanizer
    h::Int
    w::Int
    c::Int
    kernels::Vector{Matrix{Float64}}
    knames::Vector{String}
    pool::Int
    rectify::Bool
    therm::Thermometer
    names::Vector{String}

    function ConvKernel(; height::Integer, width::Integer, channels::Integer=1,
                        kernels=:edges, nthresholds::Integer=2, strategy::Symbol=:quantile,
                        pool::Integer=1, rectify::Bool=true)
        height >= 1 && width >= 1 || throw(ArgumentError("height and width must be >= 1"))
        channels >= 1 || throw(ArgumentError("channels must be >= 1"))
        pool >= 1 || throw(ArgumentError("pool must be >= 1, got $pool"))
        ks, kn = _kernel_bank(kernels)
        for k in ks
            size(k, 1) <= height && size(k, 2) <= width ||
                throw(ArgumentError("kernel $(size(k)) does not fit in $(height)x$(width) image"))
        end
        return new(Int(height), Int(width), Int(channels), ks, kn, Int(pool), rectify,
                   Thermometer(nthresholds=nthresholds, strategy=strategy), String[])
    end
end

const _SOBEL_X = Float64[-1 0 1; -2 0 2; -1 0 1]
const _SOBEL_Y = Float64[-1 -2 -1; 0 0 0; 1 2 1]
const _LAPLACIAN = Float64[0 1 0; 1 -4 1; 0 1 0]

function _kernel_bank(spec)
    spec === :edges && return ([_SOBEL_X, _SOBEL_Y, _LAPLACIAN], ["sobelx", "sobely", "lap"])
    spec === :sobel && return ([_SOBEL_X, _SOBEL_Y], ["sobelx", "sobely"])
    if spec isa AbstractVector
        isempty(spec) && throw(ArgumentError("kernel bank is empty"))
        ks = Matrix{Float64}[Matrix{Float64}(k) for k in spec]
        for k in ks
            isodd(size(k, 1)) && isodd(size(k, 2)) ||
                throw(ArgumentError("kernels must have odd side lengths, got $(size(k))"))
        end
        return (ks, ["k$i" for i in eachindex(ks)])
    end
    throw(ArgumentError("kernels must be :edges, :sobel, or a Vector of matrices; got $spec"))
end

"Response-map size for kernel `k` under valid convolution, after pooling."
function _outdims(e::ConvKernel, k::AbstractMatrix)
    rh = e.h - size(k, 1) + 1
    rw = e.w - size(k, 2) + 1
    return (cld(rh, e.pool), cld(rw, e.pool))
end

nresponses(e::ConvKernel) = e.c * sum(prod(_outdims(e, k)) for k in e.kernels)

isfitted(e::ConvKernel) = isfitted(e.therm)

function width(e::ConvKernel)
    isfitted(e) || throw(ArgumentError("ConvKernel is not fitted; width is unknown until then"))
    return width(e.therm)
end

function feature_names(e::ConvKernel)
    isfitted(e) || throw(ArgumentError("ConvKernel is not fitted; no names yet"))
    return e.names
end

"""
Valid convolution of one channel followed by max pooling, written into `out` from `off`.

Pooling windows at the right and bottom edges may be partial when `pool` does not divide the
response map. They are pooled over whatever they contain rather than dropped, so no pixel silently
stops contributing.
"""
function _respond!(out, off, e::ConvKernel, img, ch, k)
    kh, kw = size(k)
    rh, rw = e.h - kh + 1, e.w - kw + 1
    ph, pw = _outdims(e, k)
    base = (ch - 1) * e.h * e.w
    @inbounds for py in 1:ph, px in 1:pw
        best = -Inf
        for oy in ((py - 1) * e.pool + 1):min(py * e.pool, rh),
            ox in ((px - 1) * e.pool + 1):min(px * e.pool, rw)
            acc = 0.0
            for ky in 1:kh, kx in 1:kw
                acc += k[ky, kx] * img[base + (oy + ky - 2) * e.w + (ox + kx - 1)]
            end
            e.rectify && (acc = abs(acc))
            acc > best && (best = acc)
        end
        off += 1
        out[off] = best
    end
    return off
end

"Real-valued responses for every sample: `nsamples x nresponses(e)`."
function responses(e::ConvKernel, X::AbstractMatrix{<:Real})
    n, f = size(X)
    f == e.c * e.h * e.w || throw(DimensionMismatch(
        "expected $(e.c * e.h * e.w) columns for $(e.c)x$(e.h)x$(e.w), got $f"))
    R = Matrix{Float64}(undef, n, nresponses(e))
    row = Vector{Float64}(undef, nresponses(e))
    img = Vector{Float64}(undef, f)
    for i in 1:n
        @inbounds for j in 1:f
            img[j] = float(X[i, j])
        end
        off = 0
        for ch in 1:e.c, k in e.kernels
            off = _respond!(row, off, e, img, ch, k)
        end
        @inbounds for j in 1:nresponses(e)
            R[i, j] = row[j]
        end
    end
    return R
end

function _name_responses(e::ConvKernel)
    out = String[]
    for ch in 1:e.c, (ki, k) in enumerate(e.kernels)
        ph, pw = _outdims(e, k)
        for py in 1:ph, px in 1:pw
            push!(out, string("c", ch, ".", e.knames[ki], "[", py, ",", px, "]"))
        end
    end
    return out
end

function fit!(e::ConvKernel, X::AbstractMatrix{<:Real})
    isfitted(e) && throw(ArgumentError(
        "ConvKernel is already fitted. Fitting again on new data is lookahead leakage whenever " *
        "that data is not strictly earlier than everything the model will be scored on. Call " *
        "refit!(e, X) if that is genuinely what you want."))
    return _fit_unchecked!(e, X)
end

refit!(e::ConvKernel, X::AbstractMatrix{<:Real}) =
    (e.therm = Thermometer(nthresholds=e.therm.nthresholds, strategy=e.therm.strategy);
     _fit_unchecked!(e, X))

function _fit_unchecked!(e::ConvKernel, X::AbstractMatrix{<:Real})
    R = responses(e, X)
    _fit_unchecked!(e.therm, R)
    rn = _name_responses(e)
    k = e.therm.nthresholds
    thr = e.therm.thresholds::Matrix{Float64}
    e.names = [string(rn[j], ">", round(thr[j, c], sigdigits=4))
               for j in eachindex(rn) for c in 1:k]
    return e
end

function transform(e::ConvKernel, X::AbstractMatrix{<:Real})
    isfitted(e) || throw(ArgumentError("ConvKernel must be fitted before transform"))
    return transform(e.therm, responses(e, X))
end

transform(e::ConvKernel, x::AbstractVector{<:Real}) = vec(transform(e, reshape(x, 1, :)))
