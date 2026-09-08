# Is the automaton pile-up real in Hnilov's own code, and is L the cause?
#
# The claim: in a trained FPTM every included automaton sits on the include threshold, so there is no
# confidence gradient and confidence-based variants are dead on arrival. The proposed cause is `L` —
# it freezes Type Ia, the only force that can raise an already-included automaton, once a clause
# exceeds it.
#
# But that was measured only in TMCore. TMCore's feedback rules are verified identical to
# FuzzyPatternTM's state-for-state (13,078 checks in its test suite), so the dynamics ought to match;
# "ought to" is doing work in that sentence. And Hnilov's published model is compiled — automata
# discarded, only literal index lists kept — so it cannot be read directly.
#
# So: train with FuzzyPatternTM itself and read its automata. Two questions, both answered in his
# code rather than ours:
#
#   1. Does the pile-up appear at a published-style L?
#   2. Does it vanish when L is large enough that the gate never shuts?
#
# TMCore is run at the same settings alongside, so a disagreement would show up as a difference
# between implementations rather than being invisible.
#
#   julia --project=. research/pileup-verify/run.jl [epochs]

include(joinpath(@__DIR__, "..", "upstream.jl"))
include(joinpath(@__DIR__, "..", "mnist.jl"))
using Printf, Random, Statistics
using TMCore

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
const NTRAIN = 20_000          # enough to train properly, small enough to run four models
const CLAUSES, T, S, LF = 40, 10, 125, 5

const FPTM_PATH = upstream("BooBSD-FuzzyPatternTM")
include(joinpath(FPTM_PATH, "src", "FuzzyPatternTM.jl"))
const FP = Main.FuzzyPatternTM

println("="^88)
println("Is the automaton pile-up real in FuzzyPatternTM, and is L the cause?")
println("="^88)
@printf("MNIST, %d train, %d clauses/class, T %d, S %d, LF %d, %d epochs\n\n",
        NTRAIN, CLAUSES, T, S, LF, EPOCHS)

trp, tryy, ntr, _, _ = mnist_train(joinpath(REPO_ROOT, "data"))
tep, tey, nte, _, _ = mnist_test(joinpath(REPO_ROOT, "data"))
trb, _ = booleanize_both(trp, min(NTRAIN, ntr))
teb, _ = booleanize_both(tep, nte)
Ytr, Yte = Int8.(tryy[1:length(trb)]), Int8.(tey)

"Summarise the included automata: how far above the include threshold do they sit?"
function summarise(states, il)
    v = Int[]
    for M in states, x in M
        x >= il && push!(v, Int(x))
    end
    isempty(v) && return (0, 0, 0, 0.0)
    sort!(v)
    return (v[1], v[end÷2], v[end], count(>(il), v) / length(v))
end

# --------------------------------------------------------------------------
# Hnilov's implementation
# --------------------------------------------------------------------------
function run_fptm(L, epochs)
    x_train = [FP.TMInput(b) for b in trb]
    x_test = [FP.TMInput(b) for b in teb]
    tm = FP.TMClassifier{Int8}(CLAUSES, T, S; states_num=256, include_limit=128, L=L, LF=LF)
    FP.initialize!(tm, x_train, Ytr)
    for _ in 1:epochs
        FP.train!(tm, x_train, Ytr; shuffle=true)
    end
    acc = FP.accuracy(FP.predict(tm, x_test), Yte)
    states = Matrix{UInt8}[]
    il = 0
    for (_, ta) in tm.clauses
        il = Int(ta.include_limit)
        append!(states, (ta.positive_clauses, ta.negative_clauses,
                         ta.positive_clauses_inverted, ta.negative_clauses_inverted))
    end
    lits = Int[]
    for (_, ta) in tm.clauses, (l, i) in ((ta.positive_included_literals, ta.positive_included_literals_inverted),
                                          (ta.negative_included_literals, ta.negative_included_literals_inverted))
        append!(lits, [length(l[j]) + length(i[j]) for j in eachindex(l)])
    end
    return (acc=acc, spread=summarise(states, il), lit=isempty(lits) ? 0 : sort(lits)[end÷2])
end

# --------------------------------------------------------------------------
# Ours, same settings
# --------------------------------------------------------------------------
function run_tmcore(L, epochs)
    Xtr = [TMInput(Vector{Bool}(b)) for b in trb]
    Xte = [TMInput(Vector{Bool}(b)) for b in teb]
    m = TMClassifier(Int.(Ytr), 784; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(20260908)
    for _ in 1:epochs
        train!(m, Xtr, Int.(Ytr); rng=rng)
    end
    acc = accuracy(predict(m, Xte), Int.(Yte))
    states = Matrix{UInt8}[]
    il = 0
    for banks in (m.positive, m.negative), b in banks
        il = Int(b.include_limit)
        push!(states, b.state); push!(states, b.state_inv)
    end
    c = sort(literal_counts(m))
    return (acc=acc, spread=summarise(states, il), lit=c[end÷2])
end

println("implementation        L        accuracy   lits med   incl-state min/med/max   above thr")
println("-"^88)
for (L, tag) in ((10, "published-style"), (100_000, "gate never shuts"))
    for (name, f) in (("FuzzyPatternTM", run_fptm), ("TMCore", run_tmcore))
        t = @elapsed (r = f(L, EPOCHS))
        lo, med, hi, frac = r.spread
        @printf("%-16s %8d   %.4f      %5d   %4d /%4d /%4d      %5.1f%%   (%s, %.0f s)\n",
                name, L, r.acc, r.lit, lo, med, hi, 100frac, tag, t)
    end
end

println()
println("Pile-up = median sitting on the include threshold (128) with 'above thr' near zero.")
println("If it appears at L=10 in FuzzyPatternTM and vanishes at L=100000, L is the cause and the")
println("finding is about FPTM as published, not about this reimplementation.")
