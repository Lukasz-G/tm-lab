# discriminative-ranking — a readable rule out of an unreadable clause

**Status: works. A 3,443-literal clause reduces to a 10-word rule at 0.889 precision.** First
genuine interpretability result here. `julia --project=. run.jl [epochs]`; output in `results.txt`.

## Why the previous attempts failed

`imdb-readability` found the published IMDb clause is a precise, unreadable ~3,800-literal
conjunction. `literal-ranking` then tried two ways to rank those literals and both failed — automaton
confidence (which saturates) and satisfaction frequency (P ≈ 0.50, chance).

That second failure said why. **Both measure how often a literal's condition holds, not how much it
separates the classes.** An n-gram absent from nearly every review is satisfied almost always, scores
maximally, and tells you nothing.

## The score

```
score(literal) = P(satisfied | in class) - P(satisfied | out of class)
```

No automaton gradient needed, so it applies to a stock FPTM trained exactly as published. Literals
are ranked on **train** and the truncated rule evaluated on **test** — ranking on test would be
choosing the rule using the answers.

Baselines: frequency, and a random ranking as the floor.

## Result

Stock FPTM, accuracy 0.8990, one clause per class per polarity. Base rate 0.500.

| clause | top-3 | top-5 | top-10 | top-20 |
|---|---|---|---|---|
| POSITIVE, *for* (3443 literals) | 4731 docs @ **0.768** | 1186 @ **0.843** | 910 @ **0.889** | 556 @ **0.924** |
| NEGATIVE, *for* (3563 literals) | 1803 docs @ **0.832** | 180 @ **0.972** | 38 @ 0.974 | 4 @ 0.750 |
| frequency baseline | ~24950 @ 0.501 | ~24950 @ 0.501 | ~24910 @ 0.501 | ~24800 @ 0.502 |
| random baseline | ~24880 @ 0.502 | ~24760 @ 0.503 | ~24510 @ 0.505 | ~23400 @ 0.512 |

And the rules read as rules. Positive sentiment, top 10:

```
NOT "bad"        +0.232        NOT "awful"      +0.091
    "great"      +0.170        NOT "waste"      +0.089
NOT "worst"      +0.145        NOT "minutes"    +0.079
NOT "the worst"  +0.119        NOT "boring"     +0.072
    "best"       +0.113
NOT "nothing"    +0.100
```

Negative sentiment, top 10: `"bad"`, NOT `"great"`, `"no"`, `"the worst"`, `"just"`, `"plot"`,
NOT `"the best"`, NOT `"excellent"`, `"to be"`, NOT `"a great"`.

`"minutes"` and `"plot"` are the interesting ones — neither is a sentiment word, and both are
plausible markers of a review complaining about pacing or story.

## What this is, and is not

**Is:** the literals within a clause that carry the class signal, ranked, with a measured
precision/coverage curve. The extracted short rules substantially beat both baselines, and the
baselines sit at chance, so the ranking is doing the work rather than the clause selection.

**Is not:** an explanation of what the clause computes. The full clause fires on ~19,500 of 25,000
test documents; the 10-literal rule matches 910. It is a faithful high-precision *summary* of what
the clause is about, not a behavioural equivalent. Anyone quoting these as "the rule the model
learned" would be overclaiming.

**Only the positive-polarity clauses work.** The two *against* clauses produce nothing — top scores
near zero and truncations matching 0 documents. That matches `mask-mining`, where negative-clause
cores carried the class signal only 59% of the time against 92% for positive. Half the model stays
opaque.

**Precision and coverage trade off**, as they must for a conjunction: the negative clause reaches
0.974 at top-10 but by then matches only 38 documents. Useful operating points are around top-10 to
top-20 for the positive clause (910 and 556 documents at 0.889 and 0.924).

## Caveats

IMDb only, one clause count, one seed. The score is computed against the clause's own class, so it
inherits any class imbalance; IMDb is balanced, which flatters it. Whether the same works on
MNIST-style dense features — where a "literal" is a pixel and the notion of a readable rule is
weaker — has not been tried.
