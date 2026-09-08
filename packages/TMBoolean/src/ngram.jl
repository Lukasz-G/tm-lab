# N-gram presence encoding for text.
#
# The third encoder the track calls for, and the one that makes the interface's `feature_names`
# requirement obvious rather than theoretical: a TM trained on text produces clauses over literals,
# and a literal index is worthless unless it maps back to a word. Track B's IMDb work read clauses by
# hand against an ad-hoc vocabulary built inside the experiment script; this is that, made reusable
# and fitted under the same fit-once discipline as everything else.
#
# Presence, not counts. A TM literal is a bit, so a count has to be thresholded to be usable at all,
# and on documents the presence of a word carries most of what a low count would.

"""
    NGram(; n = 1, max_features = 1000, min_df = 1, lowercase = true)

Encode documents as presence bits over the most frequent n-grams of the training corpus.

Takes a `Vector{<:AbstractString}` rather than a matrix — `fit!(e, docs)` then
`transform(e, docs)` — since a corpus has no natural feature axis before the vocabulary exists.

`n` is the maximum n-gram length: `n = 2` builds unigrams *and* bigrams, because a bigram vocabulary
without its unigrams throws away the features that carry most of the signal.

`max_features` caps the vocabulary at the most document-frequent n-grams, and is the parameter that
matters most for a TM: vocabulary size *is* input width, and `s = width / S` means width silently
changes how fast clauses forget. Ties at the cutoff are broken by the n-gram string so a vocabulary
is reproducible across runs and machines.

`min_df` drops n-grams appearing in fewer than that many training documents, applied before the cap.

Tokenization is deliberately crude — split on anything that is not a letter or digit — because a
tokenizer good enough to argue about belongs behind its own interface, not buried in an encoder.
"""
mutable struct NGram <: Booleanizer
    n::Int
    max_features::Int
    min_df::Int
    lowercase::Bool
    vocab::Union{Dict{String,Int},Nothing}     # n-gram -> column
    names::Vector{String}

    function NGram(; n::Integer=1, max_features::Integer=1000, min_df::Integer=1,
                   lowercase::Bool=true)
        n >= 1 || throw(ArgumentError("n must be >= 1, got $n"))
        max_features >= 1 || throw(ArgumentError("max_features must be >= 1, got $max_features"))
        min_df >= 1 || throw(ArgumentError("min_df must be >= 1, got $min_df"))
        return new(Int(n), Int(max_features), Int(min_df), lowercase, nothing, String[])
    end
end

isfitted(e::NGram) = e.vocab !== nothing

function width(e::NGram)
    isfitted(e) || throw(ArgumentError("NGram is not fitted; width is unknown until then"))
    return length(e.vocab::Dict{String,Int})
end

function feature_names(e::NGram)
    isfitted(e) || throw(ArgumentError("NGram is not fitted; no names yet"))
    return e.names
end

function _tokens(e::NGram, doc::AbstractString)
    s = e.lowercase ? Base.lowercase(doc) : doc
    return [t for t in split(s, r"[^\p{L}\p{N}]+") if !isempty(t)]
end

"The distinct n-grams in one document, which is what document frequency counts."
function _doc_grams(e::NGram, doc::AbstractString)
    tk = _tokens(e, doc)
    out = Set{String}()
    for k in 1:e.n, i in 1:(length(tk) - k + 1)
        push!(out, join(view(tk, i:(i + k - 1)), " "))
    end
    return out
end

function fit!(e::NGram, docs::AbstractVector{<:AbstractString})
    isfitted(e) && throw(ArgumentError(
        "NGram is already fitted. Fitting a vocabulary again on new data is lookahead leakage " *
        "whenever that data is not strictly earlier than everything the model will be scored on. " *
        "Call refit!(e, docs) if that is genuinely what you want."))
    return _fit_unchecked!(e, docs)
end

refit!(e::NGram, docs::AbstractVector{<:AbstractString}) =
    (e.vocab = nothing; _fit_unchecked!(e, docs))

function _fit_unchecked!(e::NGram, docs::AbstractVector{<:AbstractString})
    isempty(docs) && throw(ArgumentError("need at least one document"))
    df = Dict{String,Int}()
    for d in docs, g in _doc_grams(e, d)
        df[g] = get(df, g, 0) + 1
    end
    kept = [g for (g, c) in df if c >= e.min_df]
    isempty(kept) && throw(ArgumentError(
        "no n-gram survives min_df = $(e.min_df) on this corpus; the vocabulary would be empty"))
    # Most frequent first, ties broken by the string itself so the vocabulary is reproducible.
    sort!(kept, by = g -> (-df[g], g))
    length(kept) > e.max_features && (kept = kept[1:e.max_features])
    e.vocab = Dict(g => i for (i, g) in enumerate(kept))
    e.names = kept
    return e
end

function transform(e::NGram, docs::AbstractVector{<:AbstractString})
    isfitted(e) || throw(ArgumentError("NGram must be fitted before transform"))
    v = e.vocab::Dict{String,Int}
    out = falses(length(docs), length(v))
    for (i, d) in enumerate(docs)
        for g in _doc_grams(e, d)
            j = get(v, g, 0)
            j != 0 && (out[i, j] = true)
        end
    end
    return out
end

transform(e::NGram, doc::AbstractString) = vec(transform(e, [doc]))
