# Chunked bit-packed input.
#
# Portions derived from Tsetlin.jl (the 64-bit chunked `TMInput` layout, which the whole clause
# evaluator is built around). Copyright (c) 2024-2026 Artem Hnilov. MIT License. See NOTICE.md.

"""
    TMInput(x::AbstractVector{Bool})

A booleanized example, packed 64 bits to a `UInt64` chunk.

Bit `i` of the example lives at bit `(i-1) % 64` of chunk `(i-1) ÷ 64 + 1`. Padding bits in the
final chunk are always zero, and clause include-masks are required to be zero there too, so padding
can never manufacture a literal miss.
"""
struct TMInput <: AbstractVector{Bool}
    chunks::Memory{UInt64}
    len::Int

    function TMInput(chunks::Memory{UInt64}, len::Int)
        length(chunks) == cld(len, 64) ||
            throw(ArgumentError("chunk count $(length(chunks)) does not match len $len"))
        return new(chunks, len)
    end
end

function TMInput(x::AbstractVector{Bool})
    len = length(x)
    chunks = Memory{UInt64}(undef, cld(len, 64))
    fill!(chunks, zero(UInt64))
    @inbounds for i in 1:len
        if x[i]
            chunks[(i - 1) >> 6 + 1] |= one(UInt64) << ((i - 1) & 63)
        end
    end
    return TMInput(chunks, len)
end

TMInput(x::AbstractArray{Bool}) = TMInput(vec(x))

Base.size(x::TMInput) = (x.len,)
Base.IndexStyle(::Type{TMInput}) = IndexLinear()

@inline function Base.getindex(x::TMInput, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds return !iszero(x.chunks[(i - 1) >> 6 + 1] & (one(UInt64) << ((i - 1) & 63)))
end

nchunks(x::TMInput) = length(x.chunks)
