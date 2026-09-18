# Benchmark harness.
#
# Deliberately conservative about what it claims. Published TM throughput headlines are usually
# batched, multithreaded, on a compiled inference-only model, and quoted as a peak; this measures
# one configuration honestly and reports the median of repeated runs rather than the best.

"""
    BenchResult

Timings from [`benchmark`](@ref). Times are per-call medians in seconds; `predictions_per_second`
is derived from the median, not from the fastest run.
"""
struct BenchResult
    nexamples::Int
    loops::Int
    predict_seconds::Float64
    predictions_per_second::Float64
    train_epoch_seconds::Union{Float64,Nothing}
    accuracy::Union{Float64,Nothing}
end

function Base.show(io::IO, r::BenchResult)
    println(io, "TMCore benchmark")
    println(io, "  examples          : ", r.nexamples, " (", r.loops, " loops)")
    println(io, "  inference         : ", round(r.predict_seconds * 1e3, digits=2), " ms per pass")
    println(io, "  throughput        : ", round(Int, r.predictions_per_second), " predictions/s")
    r.accuracy === nothing || println(io, "  accuracy          : ", round(r.accuracy, digits=4))
    r.train_epoch_seconds === nothing ||
        println(io, "  training          : ", round(r.train_epoch_seconds, digits=2), " s per epoch")
end

_median(v) = (s = sort(v); isodd(length(s)) ? s[(length(s) + 1) ÷ 2] :
                          (s[length(s) ÷ 2] + s[length(s) ÷ 2 + 1]) / 2)

"""
    benchmark(model, X, Y = nothing; loops = 5, warmup = true, train = false, rng)

Time inference over `X`, and optionally one training epoch.

`warmup` runs one untimed pass first so compilation is not counted — without it the first
measurement of a fresh session is dominated by codegen and the number is meaningless.

**Report `Threads.nthreads()` alongside any number from here.** Inference threads across examples
unconditionally, so an inference figure measured under `julia -t auto` is not comparable to one
measured at a single thread. Training follows whatever `parallel` is passed through, defaulting to
the serial schedule; see the note on `train!`.
"""
function benchmark(m::TMClassifier, X::AbstractVector{TMInput}, Y=nothing;
                   loops::Integer=5, warmup::Bool=true, train::Bool=false,
                   rng=Random.default_rng(), parallel::Symbol=:none)
    warmup && predict(m, @view X[1:min(64, length(X))])

    times = Float64[]
    local preds
    for _ in 1:loops
        t = @elapsed (preds = predict(m, X))
        push!(times, t)
    end
    t_pred = _median(times)

    acc = Y === nothing ? nothing : accuracy(preds, Y)

    t_train = nothing
    if train
        Y === nothing && throw(ArgumentError("training benchmark needs labels"))
        t_train = @elapsed train!(m, X, Y; rng=rng, parallel=parallel)
    end

    return BenchResult(length(X), loops, t_pred, length(X) / t_pred, t_train, acc)
end

"""
    model_bytes(model) -> (with_states, inference_only)

In-memory size of the clause banks, with and without the Tsetlin automata.

The gap is large and worth knowing: automata are one byte per literal *position* per clause, while
include masks are one bit, so dropping them shrinks a model by roughly a factor of `width / 8` per
clause. It is the difference between a model that fits in cache and one that does not.
"""
function model_bytes(m::TMClassifier)
    masks = states = 0
    for banks in (m.positive, m.negative), b in banks
        masks += 2 * sizeof(b.included) + sizeof(b.count)
        b.state === nothing || (states += sizeof(b.state) + sizeof(b.state_inv))
    end
    return (with_states=masks + states, inference_only=masks)
end
