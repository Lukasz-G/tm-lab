# ranking-confound — does the clause contribute anything?

**Status: mostly no. The readable words come from the data, not from the model. The
`discriminative-ranking` claim is retracted down to a much smaller one.**
`julia --project=. run.jl [epochs]`; output in `results.txt`.

## The suspicion

`discriminative-ranking` claimed a readable rule extracted from a 3,443-literal clause. But the score
used is essentially chi-square feature relevance, and the 12,800 IMDb features had **already** been
chi-square selected against the label during preparation. So the top-ranked literals might simply be
the globally most discriminative n-grams, with the clause contributing nothing.

Three rankings, identical scoring, differing only in which literals are candidates: the clause's own
literals; **all** 25,600 literals with the model ignored entirely; and a random subset as the floor.

## Result

| | top-3 | top-5 | top-10 | top-20 |
|---|---|---|---|---|
| **NEGATIVE, clause** | 1803 @ 0.832 | 180 @ 0.972 | 38 @ 0.974 | 4 @ 0.750 |
| **NEGATIVE, global** | 785 @ **0.969** | 223 @ **0.991** | 22 @ **1.000** | 0 @ — |
| **POSITIVE, clause** | 4731 @ 0.768 | 1186 @ 0.843 | 910 @ 0.889 | **556 @ 0.924** |
| **POSITIVE, global** | 4731 @ 0.768 | 1414 @ 0.836 | 154 @ **0.935** | 2 @ 1.000 |

Top-10 overlap between clause and global: **6 of 10** and **5 of 10**. And the global top-10, chosen
without the model existing, is the same vocabulary: `"bad"`, NOT `"great"`, `"worst"`, `"no"`,
`"the worst"`, `"just"`, `"plot"`.

On the positive clause the top-3 rules are **identical** — same words, same 4,731 documents, same
0.768.

## What this means

**The words are the dataset's, not the model's.** Ranking with no reference to the clause at all
finds the same vocabulary and generally *higher* precision. Anything in the earlier write-up that
read as "here is what this clause learned" was wrong; it was "here are the discriminative words in
IMDb, some of which this clause includes".

**One residual is real, and it is about coverage rather than word choice.** At larger N the
clause-restricted set stays jointly satisfiable — 556 documents at 0.924 for the positive clause —
where the global set collapses to 2 documents. Global ranking picks literals that are individually
discriminative, and their conjunction over-restricts almost immediately. Clause membership therefore
buys *co-occurrence*: the model has learned a combination of discriminative features that actually
holds together on real documents.

That is a genuine but much smaller claim than the one made. It says the clause is a useful
**selector of compatible features**, not a source of insight about which words matter.

## Consequence

`discriminative-ranking`'s headline is withdrawn. What survives is: a short, readable, high-precision
rule can be produced from IMDb, the clause helps only by keeping the selected literals mutually
satisfiable, and the interpretability question — what did *this model* learn that the data alone
does not say — remains unanswered.

The right control for any future attempt is in this directory: rank without the model and check
whether the model beat it.
