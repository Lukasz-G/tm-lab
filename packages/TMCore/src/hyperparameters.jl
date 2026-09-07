# Hyperparameters, and the validation that is not optional.
#
# Every check here guards a failure that is silent rather than loud — the model trains, predicts,
# and reports a number, and the number is meaningless. Each one cost somebody real time.

"""
    Hyperparameters(; T, S, L, LF, width)

FPTM hyperparameters, validated at construction.

- `T`  vote threshold; feedback probability saturates here.
- `S`  specificity. **A divisor, not a probability**: the per-clause forget count is
  `s = round(width / S)`, so `S` scales with input width and a value tuned at one feature count does
  not transfer to another.
- `L`  literal budget. **Gates growth, it does not cap size** — see the note below.
- `LF` allowed literal failures. `LF == 1` is the strict TM.

Rules of thumb from arXiv:2508.08350 §2.1, for orientation only:
`T ≈ sqrt(CLAUSES/2 × LF)` multiclass, `T ≈ sqrt(CLAUSES × LF)` binary.
"""
struct Hyperparameters
    T::Int
    S::Int
    L::Int
    LF::Int
    s::Int
    width::Int

    function Hyperparameters(; T::Integer, S::Integer, L::Integer, LF::Integer, width::Integer)
        # LF = 0 makes `ceiling - misses` non-positive for every clause on every input, so nothing
        # ever votes. Type Ia and Type II both gate on `vote > 0` and never fire, only the Type Ib
        # forgetting branch runs, the include set never grows, and a binary model returns the
        # negative class for every input with a decision margin of exactly zero. It reads as
        # "no signal", not as "broken" — which is why this is an error and not a warning.
        LF >= 1 || throw(ArgumentError(
            "LF must be >= 1, got $LF. LF = 0 yields a model that cannot learn: no clause can " *
            "ever vote, so the include set never grows and every decision margin is exactly zero. " *
            "The strict (classical) TM is LF = 1."))
        T >= 1 || throw(ArgumentError("T must be >= 1, got $T"))
        S >= 1 || throw(ArgumentError("S must be >= 1, got $S"))
        L >= 1 || throw(ArgumentError("L must be >= 1, got $L"))
        width >= 1 || throw(ArgumentError("width must be >= 1, got $width"))
        return new(Int(T), Int(S), Int(L), Int(LF), round(Int, width / S), Int(width))
    end
end

function Base.show(io::IO, h::Hyperparameters)
    print(io, "Hyperparameters(T=", h.T, ", S=", h.S, " (s=", h.s, "), L=", h.L,
          ", LF=", h.LF, ", width=", h.width, ")")
end

"""
    check_transfer(h, new_width)

Warn when hyperparameters tuned at one input width are reused at another. `S` is a divisor of the
width, so the effective forget rate silently changes; `T` scales with clause count and `LF`, so it
usually needs revisiting too. Returns `h` so it can be threaded through a pipeline.
"""
function check_transfer(h::Hyperparameters, new_width::Integer)
    if new_width != h.width
        s_new = round(Int, new_width / h.S)
        @warn("hyperparameters were tuned at a different input width; S is a divisor of the width, "*
              "so the per-clause forget count changes silently",
              tuned_width = h.width, new_width = new_width, s_tuned = h.s, s_new = s_new)
    end
    return h
end

"""
    alias_count(D, H, V; misses = 1)

Expected number of *other* symbols in a vocabulary of `V` whose hypervector code overlaps a given
symbol's in at least `H - misses` bits — the symbols that, on their own, drive a clause targeting
that symbol to within `misses` of a full vote.

This is the guard for running fuzzy semantics over a sparse distributed code. Below about `H = 4` a
clause one bit short of a match aliases to whole other symbols and the fuzziness stops meaning
anything; by `H = 4` at realistic `D` the alias count is under 10^-3 and the concern is gone.
Hypervector bits, not match granularity, is the lever. See `research/aliasing/`.

Assumes codes are independent uniform `H`-subsets of `[D]`. That holds for randomly assigned
property symbols; it does **not** hold for GraphTM's message symbols, which are cyclic shifts of one
another, so this is optimistic there.
"""
function alias_count(D::Integer, H::Integer, V::Integer; misses::Integer=1)
    (1 <= H <= D) || throw(ArgumentError("need 1 <= H <= D, got H=$H, D=$D"))
    V >= 1 || throw(ArgumentError("V must be >= 1, got $V"))
    den = binomial(big(D), big(H))
    tail = sum(Float64(binomial(big(H), big(j)) * binomial(big(D - H), big(H - j)) / den)
               for j in max(0, H - misses):H)
    return (V - 1) * tail
end

"""
    check_symbol_encoding(D, H, V; threshold = 0.01)

Throw when a hypervector encoding is too dense in vocabulary for bit-level fuzziness to be
meaningful. Call this at model construction whenever inputs are symbol-encoded; it turns a silent
degradation into a loud one.
"""
function check_symbol_encoding(D::Integer, H::Integer, V::Integer; threshold::Real=0.01)
    n = alias_count(D, H, V)
    n <= threshold || throw(ArgumentError(
        "hypervector encoding is too weak for fuzzy clause evaluation: with D=$D, H=$H, V=$V, " *
        "about $(round(n, sigdigits=3)) other symbols each drive a clause to within one literal " *
        "of a full match, so a near-match no longer identifies a symbol. Raise hypervector_bits " *
        "(H >= 4 is usually enough), or raise D, or use strict evaluation (LF = 1)."))
    return n
end
