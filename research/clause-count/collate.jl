# Assemble the clause-count sweep into one table.
#
# Each (clauses, T, seed) point is written by its own process into out/, so this only reads and
# averages. Run it after sweep.sh.
#
#   julia --project=. research/clause-count/collate.jl

using Printf, Statistics

const OUT = joinpath(@__DIR__, "out")

isdir(OUT) || error("no results in $OUT — run research/clause-count/sweep.sh first")

rows = Tuple{Int,Int,Int,Float64,Int,Int,Float64}[]
for f in sort(readdir(OUT))
    endswith(f, ".txt") || continue
    p = split(strip(read(joinpath(OUT, f), String)))
    length(p) == 7 || continue
    push!(rows, (parse(Int, p[1]), parse(Int, p[2]), parse(Int, p[3]), parse(Float64, p[4]),
                 parse(Int, p[5]), parse(Int, p[6]), parse(Float64, p[7])))
end
isempty(rows) && error("no parsable results in $OUT")

groups = Dict{Tuple{Int,Int},Vector{Tuple{Float64,Int,Int,Float64}}}()
for (c, t, _, a, be, ep, el) in rows
    push!(get!(groups, (c, t), []), (a, be, ep, el))
end

println("="^92)
println("CIFAR-10 clause-count sweep, best encoder (RGB x2 thermometer + fixed edge kernels)")
println("="^92)
println("T follows the FPTM paper's T ~ sqrt(CLAUSES/2 * LF); fixed-T rows are the control.\n")

# The scaled-T rows are the sweep proper; anything else is a control and is printed separately so the
# trend is not read across two different T policies.
scaled = sort([k for k in keys(groups) if k[2] == round(Int, sqrt(k[1] / 2 * 5))], by = first)
others = sort([k for k in keys(groups) if !(k in scaled)], by = first)

@printf("%9s %4s %8s %8s %10s %9s\n", "clauses", "T", "mean", "sd", "best epoch", "seconds")
println("-"^92)
# Wrapped: a top-level for loop opens a soft scope and would not assign back to `prev`.
function print_scaled(groups, scaled)
    prev = nothing
    for k in scaled
        v = groups[k]
        accs = [x[1] for x in v]
        mu = mean(accs)
        sd = length(accs) > 1 ? std(accs) : 0.0
        # Peak at the final epoch means the run had not converged and understates that arm.
        stalled = all(x -> x[2] == x[3], v) ? " *" : ""
        delta = prev === nothing ? "" : @sprintf("  %+.4f", mu - prev)
        @printf("%9d %4d %8.4f %8.4f %10.1f %9.0f%s%s\n",
                k[1], k[2], mu, sd, mean(x[2] for x in v), mean(x[4] for x in v), delta, stalled)
        prev = mu
    end
end
print_scaled(groups, scaled)

if !isempty(others)
    println()
    println("Controls (T held at its 40-clause value instead of scaled):")
    for k in others
        v = groups[k]
        accs = [x[1] for x in v]
        ref = get(groups, (k[1], round(Int, sqrt(k[1] / 2 * 5))), nothing)
        cmp = ref === nothing ? "" :
              @sprintf("   %+.4f vs scaled T", mean(accs) - mean(x[1] for x in ref))
        @printf("%9d %4d %8.4f%s\n", k[1], k[2], mean(accs), cmp)
    end
end

println()
println("* = every seed peaked on the final epoch, so that row had not converged and understates")
println("  its clause count. Read the trend, and treat a starred top row as a lower bound.")
