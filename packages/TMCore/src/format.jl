# Model container: fixed binary header, raw packed arrays, little-endian.
#
# Specified in docs/model-format.md, which is the normative description — this file must follow that
# document rather than the other way round, because the format's whole value is that someone else
# can implement it. A reference reader in pure Python lives in tools/read_tmcore.py and is checked
# against this writer on a real trained model.
#
# Explicitly NOT Julia `Serialization`: it survives neither a Julia upgrade nor a struct rename, and
# upstream renaming its central struct in 2026 invalidated every model saved before it.

const FORMAT_MAGIC = b"TMCORE\0\0"
const FORMAT_VERSION = UInt32(3)
const HEADER_FIXED_BYTES = 80

ceiling_code(::LiteralCapped) = 0x00
ceiling_code(::FlatLF) = 0x01
ceiling_from_code(c::UInt8) = c == 0x00 ? LiteralCapped() :
                              c == 0x01 ? FlatLF() :
                              throw(ArgumentError("unknown ceiling policy code $c"))

budget_code(::GrowthGate) = 0x00
budget_code(::HardCap) = 0x01
budget_from_code(c::UInt8) = c == 0x00 ? GrowthGate() :
                             c == 0x01 ? HardCap() :
                             throw(ArgumentError("unknown budget policy code $c"))

# Version 2 added the feedback policy. It affects training only, never inference, but a checkpoint
# reloaded to continue training would silently switch rules without it.
feedback_code(::ThresholdFeedback) = 0x00
feedback_code(::ProportionalFeedback) = 0x01
feedback_code(::ProportionalIdle) = 0x02
feedback_from_code(c::UInt8) = c == 0x00 ? ThresholdFeedback() :
                               c == 0x01 ? ProportionalFeedback() :
                               c == 0x02 ? ProportionalIdle() :
                               throw(ArgumentError("unknown feedback policy code $c"))

# Version 3. Unlike the feedback policy this one changes *inference*: the same clauses score
# differently under it, so a reader that ignored it would produce wrong predictions silently.
misscost_code(::UniformMissCost) = 0x00
misscost_code(::ConfidenceWeightedMissCost) = 0x01
misscost_threshold(p::ConfidenceWeightedMissCost) = p.threshold
misscost_threshold(::MissCostPolicy) = UInt16(0)
misscost_from_code(c::UInt8, thr::UInt16) =
    c == 0x00 ? UniformMissCost() :
    c == 0x01 ? ConfidenceWeightedMissCost(thr) :
    throw(ArgumentError("unknown miss-cost policy code $c"))

class_code(::Type{<:Integer}) = 0x00
class_code(::Type{String}) = 0x01
class_code(::Type{Bool}) = 0x02

state_code(::Nothing) = 0x00
state_code(::Matrix{UInt8}) = 0x01
state_code(::Matrix{UInt16}) = 0x02

function _class_block(classes::AbstractVector)
    io = IOBuffer()
    if eltype(classes) === String
        for c in classes
            b = codeunits(c)
            write(io, UInt32(length(b)))
            write(io, b)
        end
    else
        for c in classes
            write(io, Int64(c))
        end
    end
    return take!(io)
end

function _read_class_block(io::IO, nclasses::Integer, code::UInt8)
    if code == 0x01
        return [String(read(io, read(io, UInt32))) for _ in 1:nclasses]
    elseif code == 0x02
        return [read(io, Int64) != 0 for _ in 1:nclasses]
    elseif code == 0x00
        return [read(io, Int64) for _ in 1:nclasses]
    end
    throw(ArgumentError("unknown class type code $code"))
end

function _write_bank(io::IO, b::ClauseBank, with_states::Bool)
    write(io, b.included)
    write(io, b.included_inv)
    write(io, b.count)
    if with_states
        write(io, b.state)
        write(io, b.state_inv)
    end
    return nothing
end

"""
    save_model(path_or_io, model; include_states = true)

Write `model` in the TMCore format (see `docs/model-format.md`).

`include_states` keeps the Tsetlin automata, which are needed to continue training and are the bulk
of the file. Dropping them yields an inference-only model roughly `width / 8` times smaller per
clause, at the cost of being unable to train it further.
"""
function save_model(io::IO, m::TMClassifier{ClassType}; include_states::Bool=true) where {ClassType}
    b1 = m.positive[1]
    with_states = include_states && trainable(b1)
    include_states && !trainable(b1) &&
        throw(ArgumentError("model carries no automata; save with include_states = false"))

    classblock = _class_block(m.classes)
    payload = IOBuffer()
    for ci in eachindex(m.classes)
        _write_bank(payload, m.positive[ci], with_states)
        _write_bank(payload, m.negative[ci], with_states)
    end
    body = take!(payload)

    p = m.params
    write(io, FORMAT_MAGIC)
    write(io, FORMAT_VERSION)
    write(io, UInt32(HEADER_FIXED_BYTES + length(classblock)))
    write(io, UInt32(p.width), UInt32(b1.nchunks), UInt32(length(m.classes)), UInt32(b1.nclauses))
    write(io, UInt32(p.T), UInt32(p.S), UInt32(p.L), UInt32(p.LF))
    write(io, UInt32(b1.include_limit), UInt32(b1.state_min), UInt32(b1.state_max))
    write(io, ceiling_code(m.ceiling), budget_code(m.budget))
    write(io, with_states ? state_code(b1.state) : 0x00)
    write(io, class_code(ClassType))
    write(io, UInt64(length(body)))
    write(io, UInt32(length(classblock)))
    write(io, feedback_code(m.feedback))
    write(io, misscost_code(m.misscost))
    write(io, misscost_threshold(m.misscost))      # uint16; 0 means the nominal band midpoint
    write(io, classblock)
    write(io, body)
    return nothing
end

save_model(path::AbstractString, m::TMClassifier; kwargs...) =
    open(io -> save_model(io, m; kwargs...), path, "w")

"""
    load_model(path_or_io) -> TMClassifier

Read a model written by [`save_model`](@ref). Rejects an unknown format version rather than
guessing, and verifies the payload length so a truncated file fails loudly.
"""
function load_model(io::IO)
    magic = read(io, length(FORMAT_MAGIC))
    magic == FORMAT_MAGIC || throw(ArgumentError("not a TMCore model file"))
    version = read(io, UInt32)
    version == FORMAT_VERSION ||
        throw(ArgumentError("unsupported TMCore format version $version (this build reads $FORMAT_VERSION)"))
    header_size = read(io, UInt32)
    width, nchunks, nclasses, nclauses = ntuple(_ -> read(io, UInt32), 4)
    T, S, L, LF = ntuple(_ -> read(io, UInt32), 4)
    include_limit, state_min, state_max = ntuple(_ -> read(io, UInt32), 3)
    ceiling = ceiling_from_code(read(io, UInt8))
    budget = budget_from_code(read(io, UInt8))
    sbytes = read(io, UInt8)
    ctype = read(io, UInt8)
    payload_bytes = read(io, UInt64)
    classblock_bytes = read(io, UInt32)
    feedback = feedback_from_code(read(io, UInt8))
    mc_code = read(io, UInt8)
    misscost = misscost_from_code(mc_code, read(io, UInt16))
    classes = _read_class_block(io, nclasses, ctype)

    # The header is self-checking: three independently derivable quantities must agree, so a
    # truncated or mislabelled file fails here rather than as nonsense predictions later.
    HEADER_FIXED_BYTES + classblock_bytes == header_size ||
        throw(ArgumentError("header_size $(header_size) disagrees with the class block"))
    position(io) == header_size ||
        throw(ArgumentError("class block did not end at the declared payload offset"))
    nchunks == cld(width, 64) ||
        throw(ArgumentError("nchunks $(nchunks) does not match width $(width)"))

    ST = sbytes == 0x02 ? UInt16 : UInt8
    with_states = sbytes != 0x00

    per_bank = 2 * Int(nchunks) * Int(nclauses) * 8 + Int(nclauses) * 4 +
               (with_states ? 2 * Int(width) * Int(nclauses) * Int(sbytes) : 0)
    expected = 2 * Int(nclasses) * per_bank
    expected == Int(payload_bytes) ||
        throw(ArgumentError("declared payload of $(payload_bytes) bytes does not match the " *
                            "$(expected) bytes the header's dimensions imply"))

    mkbank() = begin
        inc = Matrix{UInt64}(undef, nchunks, nclauses)
        read!(io, inc)
        invm = Matrix{UInt64}(undef, nchunks, nclauses)
        read!(io, invm)
        cnt = Vector{Int32}(undef, nclauses)
        read!(io, cnt)
        st = sti = nothing
        if with_states
            st = Matrix{ST}(undef, width, nclauses)
            read!(io, st)
            sti = Matrix{ST}(undef, width, nclauses)
            read!(io, sti)
        end
        ClauseBank{ST}(Int(width), Int(nchunks), Int(nclauses), inc, invm, st, sti, cnt,
                       ST(include_limit), ST(state_min), ST(state_max))
    end

    pos = Vector{ClauseBank{ST}}(undef, nclasses)
    neg = Vector{ClauseBank{ST}}(undef, nclasses)
    for ci in 1:nclasses
        pos[ci] = mkbank()
        neg[ci] = mkbank()
    end
    eof(io) || @warn "trailing bytes after payload; file may not be what it claims"

    params = Hyperparameters(T=Int(T), S=Int(S), L=Int(L), LF=Int(LF), width=Int(width))
    CT = eltype(classes)
    return TMClassifier{CT,ST,OneVsRest,typeof(ceiling),typeof(budget),typeof(feedback),
                        typeof(misscost)}(
        params, classes, pos, neg, ceiling, budget, OneVsRest(), feedback, misscost)
end

load_model(path::AbstractString) = open(load_model, path, "r")
