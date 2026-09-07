# How far does Tsetlin.jl's flat-LF clause ceiling depart from the FPTM paper's min(n_literals, LF)?
#
# Data-free. The answer lives entirely in a trained model's literal counts, so this runs against
# Hnilov's published 40-clause MNIST model with no dataset download.
#
# Ceilings, as the three sources define them:
#   paper (arXiv:2508.08350 sec.2)  ceiling = n_literals == 0 ? LF : min(n_literals, LF)
#   FuzzyPatternTM (reference)      c = 0 < length_literals < LF ? length_literals : LF   [identical]
#   Tsetlin.jl (optimized)          max(0, LF - misses)                                   [flat LF]
#
# They coincide when a clause holds 0 or >= LF literals, and differ strictly in between.

include(joinpath(@__DIR__, "..", "upstream.jl"))
using Printf

const FPTM_PATH = upstream("BooBSD-FuzzyPatternTM")
include(joinpath(FPTM_PATH, "src", "FuzzyPatternTM.jl"))
using .FuzzyPatternTM: load

const MODEL = joinpath(FPTM_PATH, "models", "tm_optimized_40_fp.tm")

println("upstream FuzzyPatternTM @ ", upstream_commit("BooBSD-FuzzyPatternTM"))
println("upstream Tsetlin.jl     @ ", upstream_commit("BooBSD-Tsetlin.jl"))
println()

tm = load(MODEL)
LF, L = tm.LF, tm.L
@printf("model: %d clauses/class, %d classes, T=%d, S=%d, L=%d, LF=%d\n\n",
        tm.clauses_num, length(tm.clauses), tm.T, tm.S, L, LF)

paper_ceiling(n, LF) = n == 0 ? LF : min(n, LF)
flat_ceiling(_, LF)  = LF

# Collect the literal count of every clause in the model. A "clause" here is one polarity slot:
# TATeamCompiled stores positive and negative clause banks separately, each as a list of included
# literal indices plus a list of included inverted-literal indices.
counts = Int[]
for (_, ta) in sort(collect(tm.clauses), by = first)
    for (lits, inv) in ((ta.positive_included_literals, ta.positive_included_literals_inverted),
                        (ta.negative_included_literals, ta.negative_included_literals_inverted))
        for i in eachindex(lits)
            push!(counts, length(lits[i]) + length(inv[i]))
        end
    end
end

n_total  = length(counts)
n_empty  = count(==(0), counts)
n_below  = count(c -> 0 < c < LF, counts)     # the divergence band
n_atabove = count(>=(LF), counts)

println("="^70)
println("LITERALS PER CLAUSE   (", n_total, " clause slots total)")
println("="^70)
for n in 0:maximum(counts)
    k = count(==(n), counts)
    k == 0 && continue
    marker = n == 0 ? "  <- empty, both ceilings give LF" :
             n < LF ? "  <- DIVERGENCE BAND" : ""
    @printf("  %2d literals : %4d clauses  %5.1f%%%s\n", n, k, 100k / n_total, marker)
end

println()
@printf("empty (n = 0)          : %4d  %5.1f%%\n", n_empty, 100n_empty / n_total)
@printf("divergence (0 < n < LF): %4d  %5.1f%%   <- the only place the two disagree\n",
        n_below, 100n_below / n_total)
@printf("at or above LF         : %4d  %5.1f%%\n", n_atabove, 100n_atabove / n_total)

# Aggregate effect: the largest vote a class can cast if every clause matched perfectly.
paper_max = sum(paper_ceiling(c, LF) for c in counts)
flat_max  = sum(flat_ceiling(c, LF) for c in counts)
println()
println("="^70)
println("AGGREGATE CEILING  (max achievable vote mass, all clause slots)")
println("="^70)
@printf("  paper / FuzzyPatternTM : %6d\n", paper_max)
@printf("  Tsetlin.jl flat LF     : %6d\n", flat_max)
@printf("  flat over-votes by     : %6d  (%.2f%% higher)\n",
        flat_max - paper_max, 100 * (flat_max - paper_max) / paper_max)

println()
println("T = ", tm.T, ", so a class vote saturates at T long before either ceiling.")
println("The ratio above is what matters, not the absolute mass.")
