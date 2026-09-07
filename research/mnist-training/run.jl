# Stage 1 gate: does TMCore, trained from scratch, reproduce Hnilov's published FPTM behaviour?
#
# The evaluator is already known to be exact (research/tmcore-differential/) and the feedback rules
# match the reference state for state (TMCore's test suite). This closes the loop end to end: same
# hyperparameters, same data, same booleanization — does training actually get there?
#
# Two things are checked, and the second matters as much as the first.
#
#   1. Accuracy, against the published 40-clause model evaluated on the same test set.
#   2. The literal-count distribution. The published model holds 1 to 54 literals per clause under
#      L = 10, median 23. If our training reproduces that spread, the growth-gate dynamics are
#      right; if we sit near L instead, the feedback path is subtly wrong in a way that accuracy
#      alone could easily hide.
#
#   julia --project=. research/mnist-training/run.jl [epochs]

include(joinpath(@__DIR__, "..", "upstream.jl"))
include(joinpath(@__DIR__, "..", "mnist.jl"))
using Printf, Statistics, Random
using TMCore

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
const DATA = joinpath(REPO_ROOT, "data")
const WIDTH = 784

# Hyperparameters read straight off the published model, so this is like for like.
const CLAUSES, T, S, L, LF = 40, 10, 125, 10, 5

println("="^72)
println("MNIST reproduction - TMCore trained from scratch")
println("="^72)
@printf("clauses/class %d, T %d, S %d, L %d, LF %d, epochs %d\n", CLAUSES, T, S, L, LF, EPOCHS)

tr_px, tr_y, n_tr, _, _ = mnist_train(DATA)
te_px, te_y, n_te, _, _ = mnist_test(DATA)
tr_bits, _ = booleanize_both(tr_px, n_tr)
te_bits, _ = booleanize_both(te_px, n_te)
Xtr = [TMInput(Vector{Bool}(b)) for b in tr_bits]
Xte = [TMInput(Vector{Bool}(b)) for b in te_bits]
Ytr, Yte = Int.(tr_y), Int.(te_y)
@printf("train %d, test %d\n", n_tr, n_te)

# Reference point: the published model on this exact test set, scored the same way.
FPTM_PATH = upstream("BooBSD-FuzzyPatternTM")
include(joinpath(FPTM_PATH, "src", "FuzzyPatternTM.jl"))
const FPTM = Main.FuzzyPatternTM

function published_reference(te_bits, te_y)
    pub = FPTM.load(joinpath(FPTM_PATH, "models", "tm_optimized_40_fp.tm"))
    acc = mean(FPTM.predict(pub, [FPTM.TMInput(b) for b in te_bits]) .== te_y)
    counts = Int[]
    for (_, ta) in pub.clauses
        for (l, i) in ((ta.positive_included_literals, ta.positive_included_literals_inverted),
                       (ta.negative_included_literals, ta.negative_included_literals_inverted))
            append!(counts, [length(l[j]) + length(i[j]) for j in eachindex(l)])
        end
    end
    return acc, sort(counts)
end

pub_acc, pub_counts = published_reference(te_bits, te_y)
@printf("\npublished model: test accuracy %.4f, literals/clause median %d, range %d-%d\n\n",
        pub_acc, pub_counts[end÷2], pub_counts[1], pub_counts[end])

function run_training(Xtr, Ytr, Xte, Yte, epochs)
    m = TMClassifier(Ytr, WIDTH; clauses_per_class=CLAUSES, T=T, S=S, L=L, LF=LF)
    rng = MersenneTwister(20260907)
    best, best_epoch = 0.0, 0
    println("epoch   test acc     best    literals/clause (median, max)   epoch time")
    println("-"^72)
    for e in 1:epochs
        t = @elapsed train!(m, Xtr, Ytr; rng=rng)
        acc = accuracy(predict(m, Xte), Yte)
        if acc > best
            best, best_epoch = acc, e
        end
        c = sort(literal_counts(m))
        @printf("%5d   %.4f   %.4f            %4d  %4d              %6.1f s\n",
                e, acc, best, c[end÷2], c[end], t)
    end
    return m, best, best_epoch
end

m, best, best_epoch = run_training(Xtr, Ytr, Xte, Yte, EPOCHS)

c = sort(literal_counts(m))
println()
println("="^72)
@printf("best test accuracy : %.4f  (epoch %d)\n", best, best_epoch)
@printf("published model    : %.4f\n", pub_acc)
@printf("gap                : %+.4f\n", best - pub_acc)
println()
@printf("literals/clause  ours      median %3d  range %3d-%3d\n", c[end÷2], c[1], c[end])
@printf("                 published median %3d  range %3d-%3d\n",
        pub_counts[end÷2], pub_counts[1], pub_counts[end])
@printf("L = %d, so both should sit well above it if the growth gate is reproduced\n", L)
