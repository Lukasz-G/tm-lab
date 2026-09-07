# Thermometer encoding: one real feature becomes k monotone bits, `x > t1`, `x > t2`, ...
#
# Seeded from the encoder in this project's Python bridge, which already carried the fit-once
# constraint as a comment. Here it is enforced by the type instead.

"""
    Thermometer(; nthresholds = 4, strategy = :quantile)

Encode each real feature as `nthresholds` monotone bits: `x > t₁`, `x > t₂`, … with increasing `tᵢ`.
Output width is `nfeatures * nthresholds`.

Monotone rather than one-hot matters for a TM. Because the bits are nested — `x > t₃` implies
`x > t₂` — a clause can express `x > t₂` by including one literal, and an interval `t₂ < x ≤ t₃` by
including one literal and one negated literal. One-hot binning would force a disjunction, which a
conjunctive clause cannot represent at all.

`strategy`:
- `:quantile` — thresholds at interior quantiles of the training data, so bits are used about
  equally often. Robust to skew, and the sensible default.
- `:uniform` — thresholds evenly spaced between the observed min and max. Preserves the shape of the
  scale, at the cost of near-dead bits when the distribution is skewed.

Constant features get thresholds equal to their value, so every bit is `false` — which is correct,
carries no information, and is exactly what a constant feature deserves.
"""
mutable struct Thermometer <: Booleanizer
    nthresholds::Int
    strategy::Symbol
    thresholds::Union{Matrix{Float64},Nothing}   # nfeatures × nthresholds, ascending per row
    names::Vector{String}

    function Thermometer(; nthresholds::Integer=4, strategy::Symbol=:quantile)
        nthresholds >= 1 || throw(ArgumentError("nthresholds must be >= 1, got $nthresholds"))
        strategy in (:quantile, :uniform) ||
            throw(ArgumentError("strategy must be :quantile or :uniform, got :$strategy"))
        return new(Int(nthresholds), strategy, nothing, String[])
    end
end

isfitted(t::Thermometer) = t.thresholds !== nothing

function width(t::Thermometer)
    isfitted(t) || throw(ArgumentError("Thermometer is not fitted; width is unknown until then"))
    return size(t.thresholds::Matrix{Float64}, 1) * t.nthresholds
end

function feature_names(t::Thermometer)
    isfitted(t) || throw(ArgumentError("Thermometer is not fitted; no names yet"))
    return t.names
end

# Interior quantiles: for k thresholds, split at k+1 equal-mass intervals and take the k interior
# boundaries. The endpoints are excluded deliberately — a threshold at the minimum makes a bit that
# is almost always true, which costs a literal and carries nothing.
_interior(k) = [i / (k + 1) for i in 1:k]

function _quantile_sorted(v::AbstractVector{<:Real}, p::Real)
    n = length(v)
    n == 1 && return float(v[1])
    h = (n - 1) * p + 1
    lo = clamp(floor(Int, h), 1, n)
    hi = clamp(lo + 1, 1, n)
    return float(v[lo]) + (h - lo) * (float(v[hi]) - float(v[lo]))
end

function fit!(t::Thermometer, X::AbstractMatrix{<:Real})
    isfitted(t) && throw(ArgumentError(
        "Thermometer is already fitted. Fitting again on new data is lookahead leakage whenever " *
        "that data is not strictly earlier than everything the model will be scored on — it does " *
        "not error, it silently inflates the score. Call refit!(t, X) if that is genuinely what " *
        "you want."))
    return _fit_unchecked!(t, X)
end

refit!(t::Thermometer, X::AbstractMatrix{<:Real}) = (t.thresholds = nothing; _fit_unchecked!(t, X))

function _fit_unchecked!(t::Thermometer, X::AbstractMatrix{<:Real})
    n, f = size(X)
    n >= 1 || throw(ArgumentError("need at least one sample"))
    k = t.nthresholds
    thr = Matrix{Float64}(undef, f, k)
    ps = _interior(k)
    col = Vector{Float64}(undef, n)
    for j in 1:f
        @inbounds for i in 1:n
            col[i] = float(X[i, j])
        end
        if t.strategy === :quantile
            sort!(col)
            for c in 1:k
                thr[j, c] = _quantile_sorted(col, ps[c])
            end
        else
            lo, hi = extrema(col)
            for c in 1:k
                thr[j, c] = lo + (hi - lo) * ps[c]
            end
        end
    end
    t.thresholds = thr
    t.names = [string("f", j, ">", round(thr[j, c], sigdigits=4)) for j in 1:f for c in 1:k]
    return t
end

function transform(t::Thermometer, X::AbstractMatrix{<:Real})
    isfitted(t) || throw(ArgumentError("Thermometer must be fitted before transform"))
    thr = t.thresholds::Matrix{Float64}
    n, f = size(X)
    f == size(thr, 1) ||
        throw(DimensionMismatch("fitted on $(size(thr, 1)) features, got $f"))
    k = t.nthresholds
    out = falses(n, f * k)
    @inbounds for j in 1:f, c in 1:k
        col = (j - 1) * k + c
        tc = thr[j, c]
        for i in 1:n
            out[i, col] = X[i, j] > tc
        end
    end
    return out
end

# A single sample, for convenience at inference time.
transform(t::Thermometer, x::AbstractVector{<:Real}) = vec(transform(t, reshape(x, 1, :)))
