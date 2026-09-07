# Is the TMCore model format actually a format?
#
# A round-trip through our own writer proves nothing — it would pass just as well if the file were
# Julia-specific nonsense. The real question is whether a second implementation, written against the
# spec rather than against the writer, recovers the same model. So: train in Julia, save, then load
# with tools/read_tmcore.py (pure Python, standard library only) and have it independently reproduce
# Julia's per-class scores and predictions.
#
#   julia --project=. research/format-crosscheck/run.jl

include(joinpath(@__DIR__, "..", "mnist.jl"))
using Printf, Random
using TMCore

const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))   # upstream.jl is not needed here
const OUT = joinpath(@__DIR__, "artifacts")
const NCASES = 50

mkpath(OUT)

println("="^72)
println("Model format cross-check: Julia writer vs independent Python reader")
println("="^72)

tr_px, tr_y, n_tr, _, _ = mnist_train(DATA)
te_px, te_y, n_te, _, _ = mnist_test(DATA)
tr_bits, _ = booleanize_both(tr_px, 20_000)
te_bits, _ = booleanize_both(te_px, n_te)
Xtr = [TMInput(Vector{Bool}(b)) for b in tr_bits]
Xte = [TMInput(Vector{Bool}(b)) for b in te_bits]
Ytr, Yte = Int.(tr_y[1:20_000]), Int.(te_y)

# A genuinely trained model, not a fixture — a format that only handles freshly initialised models
# would hide plenty of bugs.
m = TMClassifier(Ytr, 784; clauses_per_class=40, T=10, S=125, L=10, LF=5)
rng = MersenneTwister(3)
for _ in 1:3
    train!(m, Xtr, Ytr; rng=rng)
end
acc = accuracy(predict(m, Xte), Yte)
@printf("trained a real model: test accuracy %.4f\n", acc)

# Save both ways round to exercise the states flag.
full = joinpath(OUT, "model.tmc")
lean = joinpath(OUT, "model_inference.tmc")
save_model(full, m)
save_model(lean, m; include_states=false)
@printf("saved with automata    : %s (%.1f MB)\n", basename(full), filesize(full) / 1024^2)
@printf("saved inference-only   : %s (%.1f KB, %.0fx smaller)\n",
        basename(lean), filesize(lean) / 1024, filesize(full) / filesize(lean))

# Julia-side round trip must be exact, including the policy types.
back = load_model(full)
@assert back.classes == m.classes
@assert back.params.LF == m.params.LF && back.params.L == m.params.L
@assert typeof(back.ceiling) == typeof(m.ceiling)
@assert typeof(back.budget) == typeof(m.budget)
@assert literal_counts(back) == literal_counts(m)
same = all(predict(back, Xte[i]) == predict(m, Xte[i]) for i in 1:2000)
@printf("julia round trip       : identical model, predictions agree on 2000 cases: %s\n", same)

lean_back = load_model(lean)
@assert !trainable(lean_back.positive[1])
lean_same = all(predict(lean_back, Xte[i]) == predict(m, Xte[i]) for i in 1:2000)
@printf("inference-only reload  : predicts identically without automata: %s\n", lean_same)

# Write the cases the Python side must reproduce: the raw bits, Julia's prediction, and Julia's
# per-class scores. Scores rather than just labels, so a disagreement localises to a class.
cases = joinpath(OUT, "cases.txt")
open(cases, "w") do io
    for k in 1:NCASES
        x = Xte[k]
        print(io, join(x[i] ? '1' : '0' for i in 1:784))
        print(io, " ", predict(m, x))
        for ci in 1:length(m.classes)
            print(io, " ", score(m, ci, x))
        end
        println(io)
    end
end
@printf("wrote %d cases for the Python reader to reproduce\n\n", NCASES)

py = Sys.iswindows() ? "python" : "python3"
ok = success(run(ignorestatus(`$py $(joinpath(@__DIR__, "compare.py")) $full $cases`)))

# The confidence-weighted miss cost changes *inference*, so a reader that ignored the field would be
# silently wrong rather than merely unable to resume training. That makes it the path most worth
# checking across languages, and it exercises per-automaton lookup rather than a plain popcount.
println()
println("--- confidence-weighted miss cost ---")
mw = TMClassifier(Ytr, 784; clauses_per_class=40, T=10, S=125, L=10, LF=5,
                  misscost=ConfidenceWeightedMissCost())
rngw = MersenneTwister(3)
for _ in 1:3
    train!(mw, Xtr, Ytr; rng=rngw)
end
@printf("trained: test accuracy %.4f
", accuracy(predict(mw, Xte), Yte))
fullw = joinpath(OUT, "model_weighted.tmc")
save_model(fullw, mw)
casesw = joinpath(OUT, "cases_weighted.txt")
open(casesw, "w") do io
    for k in 1:NCASES
        x = Xte[k]
        print(io, join(x[i] ? '1' : '0' for i in 1:784))
        print(io, " ", predict(mw, x))
        for ci in 1:length(mw.classes)
            print(io, " ", score(mw, ci, x))
        end
        println(io)
    end
end
okw = success(run(ignorestatus(`$py $(joinpath(@__DIR__, "compare.py")) $fullw $casesw`)))

# Scoring without automata is impossible under this policy and must fail loudly, not quietly.
leanw = joinpath(OUT, "model_weighted_lean.tmc")
save_model(leanw, mw; include_states=false)
refused = try
    predict(load_model(leanw), Xte[1]); false
catch e
    e isa ArgumentError
end
@printf("inference-only + weighted scoring refused: %s
", refused)
println()
println(ok && okw && refused && same && lean_same ?
        "PASS - an independent reader written against the spec recovers the same model." :
        "FAIL")
