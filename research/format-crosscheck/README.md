# format-crosscheck — is the model format actually a format?

**Status: PASS.** `julia --project=. run.jl`; requires Python 3 on `PATH`.

## Question

A round-trip through our own writer proves almost nothing. It would pass just as happily if the file
were Julia-shaped nonsense, because the same assumptions are baked into both ends. The only test
that means anything is whether a **second implementation, written against the specification rather
than against the writer**, recovers the same model.

That is the whole point of having a format at all. Its value is adoption by other implementations,
and "the Julia format with a Python bridge" does not get adopted — so the claim has to be checked,
not asserted.

## Setup

Train a real model in Julia (MNIST, 40 clauses/class, 3 epochs, 0.94 test accuracy — a trained model
rather than a fixture, since a format that only handles freshly initialised models hides plenty of
bugs). Save it. Then load it with [`tools/read_tmcore.py`](../../tools/read_tmcore.py) — pure Python,
standard library only, no NumPy, no Julia — and have it independently compute per-class scores and
predictions from the spec's formula.

Scores are compared rather than only labels, so a disagreement localises to a class instead of
hiding behind a lucky argmax.

## Result

```
trained a real model: test accuracy 0.9400
saved with automata    : model.tmc (0.7 MB)
saved inference-only   : model_inference.tmc (83.0 KB, 8x smaller)
julia round trip       : identical model, predictions agree on 2000 cases: true
inference-only reload  : predicts identically without automata: true

python: loaded 10 classes, 20 clauses/polarity, width 784, LF 5, ceiling=literal-capped
python: 50 cases, 0 score mismatches, 0 prediction mismatches
```

Every per-class score matches exactly, from an independent reader. The format is readable by someone
who has only the document.

## What this exercises that a round trip would not

- **Chunk padding.** Width 784 is not a multiple of 64, so the final chunk carries 16 bits of
  padding. Both implementations have to agree that those bits are zero and mean nothing.
- **Array ordering.** Julia writes column-major; the Python reader reshapes clause-major from the
  spec's wording alone. A transposed read would produce plausible-looking garbage.
- **The ceiling policy.** It is stored as a code, and the Python reader applies the
  literal-capped rule from the spec. Reading it as flat `LF` would change scores on exactly the
  clauses holding fewer than `LF` literals — a small, easily-missed set.
- **The stored literal count.** The reader recomputes it by popcount and refuses to load a file
  where the two disagree, so a writer bug cannot pass silently.

## Also checked

The inference-only variant drops the Tsetlin automata and predicts identically at **8x smaller**
(83 KB against 695 KB). Automata are a byte per literal position per clause where include masks are
a bit, so the gap is structural rather than incidental. A model reloaded *with* automata continues
training correctly, which is the property that makes checkpointing meaningful.

The format also refuses what it cannot trust: wrong magic, an unknown version, a truncated payload,
and a header whose declared dimensions disagree with themselves. Each is a test rather than a
comment.
