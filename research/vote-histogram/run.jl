# Is FPTM's fuzziness load-bearing, or decorative?
#
# A fuzzy clause votes an integer in [0, ceiling]. If the vote distribution is bimodal at 0 and at
# the ceiling, the clause is behaving like a strict TM clause with extra steps and LF is decorative.
# If the mass is spread across the interior, fuzziness is doing real work and the clause is an
# m-of-n rule whose firing structure is worth recovering.
#
# Runs against Hnilov's published 40-clause MNIST model, so this measures FPTM as published rather
# than as reimplemented. MNIST test set is fetched directly (no MLDatasets dependency) into
# data/, which is gitignored.

include(joinpath(@__DIR__, "..", "upstream.jl"))
using Printf, Statistics

const FPTM_PATH = upstream("BooBSD-FuzzyPatternTM")
include(joinpath(FPTM_PATH, "src", "FuzzyPatternTM.jl"))
using .FuzzyPatternTM: load, TMInput, predict

const MODEL = joinpath(FPTM_PATH, "models", "tm_optimized_40_fp.tm")
const DATA = joinpath(REPO_ROOT, "data")

# --------------------------------------------------------------------------
# MNIST test set, straight from the idx files.
# --------------------------------------------------------------------------
const MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"

function fetch_idx(name)
    mkpath(DATA)
    gz, raw = joinpath(DATA, name * ".gz"), joinpath(DATA, name)
    if !isfile(raw)
        isfile(gz) || download("$MIRROR/$name.gz", gz)
        open(raw, "w") do out
            write(out, read(pipeline(`gzip -dc $gz`)))
        end
    end
    return read(raw)
end

be32(b, i) = (Int(b[i]) << 24) | (Int(b[i+1]) << 16) | (Int(b[i+2]) << 8) | Int(b[i+3])

function mnist_test()
    ib = fetch_idx("t10k-images-idx3-ubyte")
    lb = fetch_idx("t10k-labels-idx1-ubyte")
    be32(ib, 1) == 2051 || error("bad image magic")
    be32(lb, 1) == 2049 || error("bad label magic")
    n, nr, nc = be32(ib, 5), be32(ib, 9), be32(ib, 13)
    px = reshape(ib[17:16+n*nr*nc], nc, nr, n)      # idx is row-major; this lands transposed
    labels = Int8.(lb[9:8+n])
    return px, labels, n, nr, nc
end

px, y_test, n_img, nr, nc = mnist_test()
@printf("MNIST test: %d images, %dx%d\n", n_img, nr, nc)

# booleanize(x, 0.25) upstream is `x .> 0.25` on Float32 in [0,1], i.e. byte > 63.75.
# Orientation matters: the model was trained on MLDatasets' layout, and getting it wrong permutes
# every bit. Decide it by accuracy rather than by assumption.
bits_raw   = [BitVector(vec(view(px, :, :, i)) .> 0x3f) for i in 1:n_img]
bits_trans = [BitVector(vec(permutedims(view(px, :, :, i))) .> 0x3f) for i in 1:n_img]

tm = load(MODEL)
LF = tm.LF
acc(bv) = mean(predict(tm, [TMInput(b) for b in bv[1:2000]]) .== y_test[1:2000])

a_raw, a_trans = acc(bits_raw), acc(bits_trans)
@printf("orientation check: raw %.4f, transposed %.4f\n", a_raw, a_trans)
bits = a_trans > a_raw ? bits_trans : bits_raw
chosen = a_trans > a_raw ? "transposed" : "raw"
best = max(a_raw, a_trans)
best < 0.90 && error("neither orientation reaches 90% ($best) — booleanization does not match the model")
@printf("using %s orientation, accuracy on 2000 test images = %.4f\n\n", chosen, best)

inputs = [TMInput(b) for b in bits]

# --------------------------------------------------------------------------
# Per-clause vote distribution.
# --------------------------------------------------------------------------
# Reimplemented rather than called, so the satisfied mask can be recorded in the same pass. This is
# the paper's / FuzzyPatternTM's ceiling: 0 < n < LF ? n : LF.
function clause_vote!(satisfied, x::BitVector, lits, inv, LF)
    n = length(lits) + length(inv)
    c = (0 < n < LF) ? n : LF
    k = 0
    @inbounds for i in eachindex(lits)
        k += 1
        hit = x[lits[i]]
        satisfied[k] += hit
        c -= !hit
    end
    @inbounds for i in eachindex(inv)
        k += 1
        hit = !x[inv[i]]
        satisfied[k] += hit
        c -= !hit
    end
    return max(c, 0)
end

ceiling_of(n, LF) = (0 < n < LF) ? n : LF

function analyze(tm, bits, LF, n_img)
    hist = Dict{Int,Int}()             # vote value -> count, pooled over all clause slots
    per_clause_hist = Vector{Vector{Int}}()
    sat_spread = Float64[]             # per clause: spread of literal satisfaction frequencies
    n_slots = 0
    for (_, ta) in sort(collect(tm.clauses), by = first)
        for (lits_v, inv_v) in ((ta.positive_included_literals, ta.positive_included_literals_inverted),
                                (ta.negative_included_literals, ta.negative_included_literals_inverted))
            for j in eachindex(lits_v)
                lits, inv = lits_v[j], inv_v[j]
                nlit = length(lits) + length(inv)
                h = zeros(Int, ceiling_of(nlit, LF) + 1)
                satisfied = zeros(Int, max(nlit, 1))
                for x in bits
                    v = clause_vote!(satisfied, x, lits, inv, LF)
                    h[v+1] += 1
                    hist[v] = get(hist, v, 0) + 1
                end
                push!(per_clause_hist, h)
                if nlit > 0
                    f = satisfied ./ n_img
                    push!(sat_spread, maximum(f) - minimum(f))
                end
                n_slots += 1
            end
        end
    end
    return hist, per_clause_hist, sat_spread, n_slots
end

hist, per_clause_hist, sat_spread, n_slots = analyze(tm, bits, LF, n_img)
total = n_slots * n_img
println("="^70)
println("POOLED VOTE DISTRIBUTION  ($n_slots clause slots x $n_img inputs)")
println("="^70)
for v in 0:LF
    k = get(hist, v, 0)
    bar = "#"^round(Int, 60 * k / total)
    @printf("  vote %d : %9d  %6.2f%%  %s\n", v, k, 100k / total, bar)
end

nonzero = total - get(hist, 0, 0)
at_ceiling = get(hist, LF, 0)
interior = nonzero - at_ceiling
println()
@printf("nonzero votes            : %9d  %6.2f%% of all evaluations\n", nonzero, 100nonzero / total)
@printf("  ...at the ceiling (=%d) : %9d  %6.2f%% of nonzero\n", LF, at_ceiling, 100at_ceiling / max(nonzero, 1))
@printf("  ...strictly interior    : %9d  %6.2f%% of nonzero  <- the load-bearing fraction\n",
        interior, 100interior / max(nonzero, 1))

println()
println("="^70)
println("PER-CLAUSE SHAPE")
println("="^70)
frac_interior = Float64[]
for h in per_clause_hist
    tot = sum(h); nz = tot - h[1]
    push!(frac_interior, nz == 0 ? 0.0 : (nz - h[end]) / nz)
end
@printf("interior fraction per clause: median %.3f, mean %.3f, min %.3f, max %.3f\n",
        median(frac_interior), mean(frac_interior), minimum(frac_interior), maximum(frac_interior))
@printf("clauses that are effectively strict (interior < 5%%): %d of %d\n",
        count(<(0.05), frac_interior), length(frac_interior))
@printf("literal satisfaction spread (max-min freq): median %.3f, min %.3f\n",
        median(sat_spread), minimum(sat_spread))
println()
println("Low spread means every included literal matches about equally often, so the satisfied")
println("mask carries little structure. High spread means some literals are near-always on and")
println("the clause has a recoverable core plus a tail.")
