# message-round — Stage 4: one message round

**Status: the machinery works, on a task built so it has to. But the messages here are the
degenerate kind, and the interesting case is untested.**
`julia --project=. run.jl [epochs] [seeds]`; output in `results.txt`.

## Task, and why it is synthetic on purpose

The design notes' benchmark trap: on Amazon Sales, flat FPTM beats GraphTM at every noise level, so
that comparison says nothing about whether topology contributes. So this task is constructed to make
a flat model provably unable to win.

Sequences over 8 symbols, length 12. Label = whether symbols 1 and 2 appear **adjacently**. Negatives
contain both symbols but never adjacent, so the classes match on symbol *presence* — measured
difference 0.0149 — and only adjacency separates them.

## Result — 20 epochs, 3 seeds, 40 clauses/class

| arm | width | accuracy |
|---|---|---|
| flat, bag of symbols **[control]** | 8 | **0.5156** |
| flat, positional one-hot | 96 | 0.9877 |
| per-node, no messages **[control]** | 8 | **0.5156** |
| **per-node, one message round** | 24 | **1.0000** |

**Both controls sit at chance**, which is what makes the rest readable. A bag of symbols cannot see
adjacency, and a node that sees only its own symbol cannot see a pair — as constructed.

One message round solves the task **perfectly at a quarter of the input width** of the flat
positional encoding, which itself only reaches 0.9877 despite having every position available.

## The honest qualification

The messages here are **identity messages**: a node receives its neighbours' symbols. That is what
one round reduces to when the message-producing clause is untrained, and it is functionally a
3-gram feature window. Calling it message passing is technically accurate and rhetorically
generous.

What distinguishes GraphTM is **learned** messages — a clause set whose outputs become the symbols
passed along edges, trained jointly with the classifier. That is Stage 5 and is untested. Nothing
here shows that learned messages beat a fixed neighbourhood window, which is the question that
actually matters.

So this establishes that the per-node evaluator, the node aggregation and the edge structure work
end to end on a task with a known answer. It does not establish that graph structure earns its
keep.

## Caveats

Synthetic task chosen to reward adjacency; not evidence about real problems. Aggregation is `max`
over nodes and credit assignment is uniform over firing nodes, both following
[`convolutional/`](../convolutional/). One sequence length, one alphabet size.
