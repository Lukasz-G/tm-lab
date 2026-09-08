# Does an FPTM class vote actually carry CLAUSES x LF resolution?
#
# The case for FPTM regression, and the reason the design notes call it the strongest standalone
# opportunity in this stack, is an arithmetic one. A Regression TM's output resolution is bounded by
# its clause count, because each clause contributes 0 or 1. FPTM clause outputs are integers in
# [0, LF], so the vote sum should range over CLAUSES x LF and buy fine-grained regression at a
# fraction of the clauses.
#
# That argument assumes the votes spread. `graded-messages` measured them concentrating at 1.1-1.6
# out of an available 5, which if general would cut the claimed advantage to a fraction of LF. This
# script checks it directly, and cheaply, BEFORE anyone writes a third feedback rule: RTM's
# error-magnitude feedback is a separate spec to reconcile and is not worth building against an
# arithmetic claim that has not been checked.
#
# WHAT IS MEASURED, on trained flat classifiers:
#
#   effective resolution   exp(H) of the class-vote distribution, i.e. the number of equally-likely
#                          distinct vote values the sum behaves like. Entropy rather than a raw
#                          distinct count, because a value hit twice in 10,000 inputs is not
#                          resolution anyone can regress on.
#
#   the two bounds         CLAUSES (what an RTM gets, one bit per clause) and CLAUSES x LF (what the
#                          FPTM regression argument claims). The question is which one the measured
#                          number sits near.
#
#   binarized control      the same model with each clause output clamped to 0/1, which is what an
#                          RTM-style clause would contribute. This is the arm that matters: if the
#                          fuzzy vote resolves no better than its own binarization, the extra
#                          resolution is arithmetic that never reaches the output.
#
# The prediction from `graded-messages` is that effective resolution lands far below CLAUSES x LF.
# A result near CLAUSES x LF refutes that and makes regression worth building; a result near CLAUSES
# says the fuzzy advantage for regression is mostly notional.
#
#   julia --project=. research/vote-resolution/run.jl [epochs]

using Printf, Random, Statistics
using TMCore

include(joinpath(@__DIR__, "..", "mnist.jl"))

const EPOCHS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
const DATA = normpath(joinpath(@__DIR__, "..", "..", "data"))
const WIDTH = 784

"Shannon entropy of the empirical distribution of `xs`, in nats."
function entropy(xs)
    c = Dict{Int,Int}()
    for x in xs; c[x] = get(c, x, 0) + 1; end
    n = length(xs)
    -sum(v / n * log(v / n) for v in values(c))
end

"""
Class-vote sums for every test input, under the model's own ceiling and under a binarized clause
output. Returns one vector per mode.

The binarized arm is the control the whole script rests on, so it reuses the same trained model and
the same clause evaluations, differing only in `min(v, 1)`. Training a separate strict model instead
would confound the comparison with everything else that differs between two training runs.
"""
function vote_sums(m, X, ci)
    LF, cp = m.params.LF, m.ceiling
    fuzzy = Int[]; strict = Int[]
    pos, neg = m.positive[ci], m.negative[ci]
    for x in X
        f = s = 0
        @inbounds for j in 1:pos.nclauses
            v = clause_vote(pos, j, x, LF, cp); f += v; s += min(v, 1)
        end
        @inbounds for j in 1:neg.nclauses
            v = clause_vote(neg, j, x, LF, cp); f -= v; s -= min(v, 1)
        end
        push!(fuzzy, f); push!(strict, s)
    end
    return fuzzy, strict
end

function main()
    tr_px, tr_y, n_tr, _, _ = mnist_train(DATA)
    te_px, te_y, n_te, _, _ = mnist_test(DATA)
    tr_bits, _ = booleanize_both(tr_px, n_tr)
    te_bits, _ = booleanize_both(te_px, n_te)
    Xtr = [TMInput(Vector{Bool}(b)) for b in tr_bits]
    Xte = [TMInput(Vector{Bool}(b)) for b in te_bits]
    ytr, yte = Int.(tr_y), Int.(te_y)

    println("="^92)
    println("Does an FPTM class vote carry CLAUSES x LF resolution, or only CLAUSES?")
    println("="^92)
    @printf("MNIST, %d train, %d test, %d epochs\n\n", length(ytr), length(yte), EPOCHS)

    # Swept because the whole claim is about how resolution scales: if the fuzzy advantage is real it
    # should hold as LF grows, and a single LF could not tell a constant apart from a trend.
    for (nclauses, LF) in ((40, 5), (40, 10), (200, 5), (200, 10))
        m = TMClassifier(ytr, WIDTH; clauses_per_class=nclauses, T=8, S=40, L=10, LF=LF)
        rng = MersenneTwister(20260908)
        for _ in 1:EPOCHS
            train!(m, Xtr, ytr; rng=rng)
        end
        acc = accuracy(predict(m, Xte), yte)

        fz, st = vote_sums(m, Xte, 1)
        rf, rs = exp(entropy(fz)), exp(entropy(st))
        mv = mean(filter(>(0), [clause_vote(m.positive[1], j, x, m.params.LF, m.ceiling)
                                for x in Xte[1:500] for j in 1:m.positive[1].nclauses]))

        # Everything is reported as a fuzzy-over-binarized ratio against LF. Comparing exp(H) to
        # CLAUSES or CLAUSES x LF directly would be apples to oranges: those are bounds on the
        # *range* a vote sum can take, and no model of either kind attains its own bound's worth of
        # entropy. The binarized arm is the like-for-like RTM proxy, and LF is the factor the
        # regression argument says separates the two.
        rgf = maximum(fz) - minimum(fz)
        rgs = maximum(st) - minimum(st)
        println("-"^92)
        @printf("%d clauses/class, LF %2d   test accuracy %.4f\n", nclauses, LF, acc)
        @printf("  vote-sum range        fuzzy %5d   binarized %5d   ratio %5.2fx   claimed %dx\n",
                rgf, rgs, rgf / rgs, LF)
        @printf("  effective, exp(H)     fuzzy %5.1f   binarized %5.1f   ratio %5.2fx   claimed %dx\n",
                rf, rs, rf / rs, LF)
        @printf("  mean firing clause vote     %5.2f   of ceiling %d\n", mv, LF)
    end

    println()
    println("="^92)
    println("Compare the measured ratio against the claimed one in each row. The ratio is what the")
    println("fuzzy vote actually buys over its own binarization; LF is what the regression argument")
    println("assumes it buys. The last line of each block is why they differ: firing votes sit near")
    println("the bottom of [1, LF] instead of spreading across it.")
end

main()
