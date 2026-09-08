"""
    TMBoolean

Booleanization for Tsetlin Machines: turning real-valued or structured input into the bit vectors a
TM consumes.

**Deliberately depends on nothing, not even TMCore.** It emits plain `BitMatrix`, so any TM
implementation in any language can adopt the encoders or port them without adopting this stack. That
is the whole argument for the track — a booleanization library is only worth writing if it is usable
in isolation.

**Why this is not a utility module.** A large share of reported TM accuracy lives here rather than in
the machine. FPTM's Fashion-MNIST state-of-the-art leans on fixed convolutional kernels applied at
booleanization, and the paper explicitly disclaims those as not part of the TM. If the encoder is
doing the work, then comparisons between TM variants that vary the encoder are not comparing TMs.
So the encoders are first-class, named, and measured.

Two properties every encoder here must have:

  * **`fit!` then `transform`, never both at once by accident.** Thresholds are fitted on training
    data only. Refitting on data that includes the future leaks lookahead — silently, and in a
    walk-forward setting fatally. `fit!` on an already-fitted encoder throws rather than obliges.
  * **`feature_names`.** Track B established that recovering a rule means mapping literal 4,412 back
    to something meaningful. An encoder that cannot say what its bits mean makes every model built
    on it unreadable, so naming is part of the interface rather than a debugging aid.
"""
module TMBoolean

export Booleanizer, Thermometer, ConvKernel, NGram,
       fit!, refit!, transform, fit_transform!, isfitted, width, feature_names,
       responses, nresponses

"""
    Booleanizer

Base type for encoders. Implementations provide [`fit!`](@ref), [`transform`](@ref),
[`width`](@ref) and [`feature_names`](@ref).
"""
abstract type Booleanizer end

"""
    isfitted(b) -> Bool

Whether the encoder has been fitted and can transform.
"""
function isfitted end

"""
    width(b) -> Int

Number of output bits per sample. Errors if the encoder is not yet fitted, since the width generally
depends on the fitted data's shape.
"""
function width end

"""
    feature_names(b) -> Vector{String}

One name per output bit, in bit order, so a literal index can be read back as a condition.
"""
function feature_names end

"""
    fit!(b, X) -> b

Fit on `X`, a `nsamples x nfeatures` matrix. Throws if `b` is already fitted; see
[`refit!`](@ref) for the deliberate override.
"""
function fit! end

"""
    transform(b, X) -> BitMatrix

Encode `X` as `nsamples x width(b)` bits.
"""
function transform end

"""
    refit!(b, X) -> b

Discard a previous fit and fit again.

Exists so that refitting is possible but never accidental. Refitting on data that includes samples
later than the ones a model will be evaluated on is lookahead leakage: it does not error, it does not
warn, it just quietly inflates the score. Requiring a different function name is the cheapest
available guard.
"""
function refit! end

"""
    fit_transform!(b, X) -> BitMatrix

Fit and transform in one call. Correct on training data and wrong on anything else — prefer `fit!`
on train followed by `transform` on train and test separately, which makes the asymmetry visible at
the call site.
"""
fit_transform!(b::Booleanizer, X) = (fit!(b, X); transform(b, X))

include("thermometer.jl")
include("convkernel.jl")
include("ngram.jl")

end # module TMBoolean
