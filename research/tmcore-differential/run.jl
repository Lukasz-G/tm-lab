# Does TMCore's bit-packed evaluator agree with FuzzyPatternTM, exactly?
#
# TMCore stores clauses as packed 64-bit include masks and counts misses with a branch-free kernel
# plus popcount. FuzzyPatternTM stores them as two lists of literal indices and walks them. Same
# semantics, entirely different mechanics — so agreement is a real check rather than a tautology,
# and disagreement localises to whichever clause and input first diverge.
#
# Run against Hnilov's published 40-clause MNIST model, on the full 10,000-image test set:
# 400 clause slots x 10,000 inputs = 4,000,000 comparisons per ceiling policy.
#
#   julia --project=. research/tmcore-differential/run.jl

include(joinpath(@__DIR__, "..", "upstream.jl"))
include(joinpath(@__DIR__, "..", "mnist.jl"))
using Printf, Statistics
using TMCore

const FPTM_PATH = upstream("BooBSD-FuzzyPatternTM")
include(joinpath(FPTM_PATH, "src", "FuzzyPatternTM.jl"))
const FPTM = Main.FuzzyPatternTM

const MODEL = joinpath(FPTM_PATH, "models", "tm_optimized_40_fp.tm")
const WIDTH = 784                       # MNIST 28x28, one booleanization threshold

println("upstream FuzzyPatternTM @ ", upstream_commit("BooBSD-FuzzyPatternTM"))

tm = FPTM.load(MODEL)
LF = tm.LF
@printf("model: %d clauses/class, %d classes, L=%d, LF=%d\n", tm.clauses_num, length(tm.clauses), tm.L, LF)

# --------------------------------------------------------------------------
# Data, with the orientation decided by accuracy rather than assumption.
# --------------------------------------------------------------------------
px, y_test, n_img, _, _ = mnist_test(joinpath(REPO_ROOT, "data"))
raw, trans = booleanize_both(px, n_img)
acc(bv) = mean(FPTM.predict(tm, [FPTM.TMInput(b) for b in bv[1:2000]]) .== y_test[1:2000])
a_raw, a_tr = acc(raw), acc(trans)
bits = a_raw >= a_tr ? raw : trans
@printf("orientation: raw %.4f vs transposed %.4f -> using %s\n\n",
        a_raw, a_tr, a_raw >= a_tr ? "raw" : "transposed")
max(a_raw, a_tr) < 0.90 && error("booleanization does not match the model")

fptm_inputs = [FPTM.TMInput(b) for b in bits]
core_inputs = [TMCore.TMInput(Vector{Bool}(b)) for b in bits]

# --------------------------------------------------------------------------
# Import every clause slot into a TMCore ClauseBank.
# --------------------------------------------------------------------------
# Each class holds a positive and a negative bank; both are imported so nothing is sampled.
banks = Tuple{ClauseBank,Vector{Vector{UInt16}},Vector{Vector{UInt16}}}[]
for (_, ta) in sort(collect(tm.clauses), by = first)
    for (lits, inv) in ((ta.positive_included_literals, ta.positive_included_literals_inverted),
                        (ta.negative_included_literals, ta.negative_included_literals_inverted))
        push!(banks, (ClauseBank(WIDTH, lits, inv), lits, inv))
    end
end
n_slots = sum(length(b[1]) for b in banks)
@printf("imported %d banks, %d clause slots\n", length(banks), n_slots)

# Import must be lossless: the packed masks have to round-trip back to the same index sets.
for (bank, lits, inv) in banks, j in 1:length(bank)
    got_l = Int[]; got_i = Int[]
    for n in 1:bank.nchunks, b in 0:63
        i = (n - 1) * 64 + b + 1
        i > WIDTH && break
        iszero(bank.included[n, j] & (one(UInt64) << b))     || push!(got_l, i)
        iszero(bank.included_inv[n, j] & (one(UInt64) << b)) || push!(got_i, i)
    end
    (got_l == sort(Int.(lits[j])) && got_i == sort(Int.(inv[j]))) ||
        error("import is lossy at clause $j")
end
println("import round-trips losslessly\n")

# --------------------------------------------------------------------------
# The differential itself.
# --------------------------------------------------------------------------
function differential(banks, fptm_inputs, core_inputs, LF, policy)
    mismatches = 0
    first_bad = nothing
    checked = 0
    sat = UInt64[]
    for (bank, lits, inv) in banks
        length(sat) == bank.nchunks || (sat = zeros(UInt64, bank.nchunks))
        for j in 1:length(bank)
            for k in eachindex(core_inputs)
                want = FPTM.check_clause(fptm_inputs[k], lits[j], inv[j], LF)
                got = clause_vote!(sat, bank, j, core_inputs[k], LF, policy)
                checked += 1
                if want != got
                    mismatches += 1
                    first_bad === nothing && (first_bad = (j, k, want, got))
                end
            end
        end
    end
    return checked, mismatches, first_bad
end

# LiteralCapped is what FuzzyPatternTM implements, so this pair must agree exactly.
checked, mism, bad = differential(banks, fptm_inputs, core_inputs, LF, LiteralCapped())
@printf("LiteralCapped vs FuzzyPatternTM : %d comparisons, %d mismatches\n", checked, mism)
bad === nothing || @printf("  first divergence: clause %d, input %d, want %d, got %d\n", bad...)

# FlatLF is Tsetlin.jl's semantics, so it must NOT agree everywhere — it should differ exactly on
# the clauses holding between 1 and LF-1 literals, and nowhere else. A silent match would mean the
# policy parameter is not actually reaching the evaluator.
_, mism_flat, _ = differential(banks, fptm_inputs, core_inputs, LF, FlatLF())
band = sum(count(j -> 0 < bank.count[j] < LF, 1:length(bank)) for (bank, _, _) in banks)
@printf("FlatLF vs FuzzyPatternTM        : %d mismatches, from %d clause slots in the divergence band\n",
        mism_flat, band)

# --------------------------------------------------------------------------
# The satisfied mask has to mean what it claims to mean.
# --------------------------------------------------------------------------
function check_masks(banks, core_inputs, bits, LF)
    bad = 0
    for (bank, lits, inv) in banks
        sat = zeros(UInt64, bank.nchunks)
        for j in 1:min(length(bank), 4)              # a sample: this check is O(width) per input
            for k in 1:200
                clause_vote!(sat, bank, j, core_inputs[k], LF, LiteralCapped())
                x = bits[k]
                want = sort(vcat(Int[i for i in lits[j] if x[i]], Int[i for i in inv[j] if !x[i]]))
                got = Int[]
                for n in 1:bank.nchunks, b in 0:63
                    i = (n - 1) * 64 + b + 1
                    i > WIDTH && break
                    iszero(sat[n] & (one(UInt64) << b)) || push!(got, i)
                end
                want == got || (bad += 1)
            end
        end
    end
    return bad
end
mask_bad = check_masks(banks, core_inputs, bits, LF)
@printf("satisfied mask                  : %d disagreements with the naive index recomputation\n", mask_bad)

println()
if mism == 0 && mask_bad == 0 && mism_flat > 0
    println("PASS - the packed evaluator reproduces the reference exactly, the satisfied mask is")
    println("       faithful, and the ceiling policy demonstrably changes the result.")
else
    println("FAIL")
end
