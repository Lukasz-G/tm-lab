# vote-resolution

**Q.** The case for FPTM regression is arithmetic. An RTM's output resolution is bounded by its
clause count, because each clause contributes 0 or 1; FPTM clause outputs are integers in [0, `LF`],
so the vote sum should span `CLAUSES × LF` and buy fine-grained regression at a fraction of the
clauses. [`graded-messages`](../graded-messages/) measured firing votes concentrating at 1.1–1.6 out
of an available 5, which would cut that advantage badly. Does the resolution exist?

Measured on trained MNIST classifiers, against the same model with each clause output clamped to 0/1
— the like-for-like RTM proxy. Comparing to `CLAUSES × LF` directly would be apples to oranges:
that is a bound on the *range*, and no model of either kind attains its own bound's worth of entropy.

| clauses/class | `LF` | range ratio | effective ratio, exp(H) | **claimed** | mean firing vote |
|---|---|---|---|---|---|
| 40 | 5 | 2.93× | 2.52× | **5×** | 2.03 of 5 |
| 40 | 10 | 3.68× | 3.40× | **10×** | 2.56 of 10 |
| 200 | 5 | 2.45× | 2.23× | **5×** | 1.71 of 5 |
| 200 | 10 | 3.18× | 2.90× | **10×** | 1.80 of 10 |

**A. The advantage is real but roughly a third of the claim, and it does not scale with `LF`.**

Doubling `LF` from 5 to 10 does not double resolution — it moves 2.52× to 3.40× at 40 clauses, and
2.23× to 2.90× at 200. The claimed relation is linear in `LF`; the measured one is sublinear and
flattening near 3×. The mechanism is on the last column: raising the ceiling barely raises the votes
(1.71 → 1.80 at 200 clauses). A clause's vote is set by how completely its pattern matches, not by
how much headroom `LF` gives it, so most of a large `LF` is never used.

**Why this was worth running first.** RTM's error-magnitude feedback is a third feedback spec to
reconcile against the two already implemented, with no reference implementation to test against. It
is not worth building to chase a factor that had never been checked. The finding does not kill the
regression track — a 3× clause reduction at equal resolution is still worth having — but it sets the
expectation at 3× rather than `LF`×, and it says the obvious knob for improving it, turning `LF` up,
does not work.

**Caveats.** MNIST only, one booleanization, `T`/`S`/`L` fixed, and resolution measured on a
classifier's vote sums rather than on a trained regressor. A regression training loop puts different
pressure on the vote distribution than one-vs-rest classification does, and could plausibly spread it
further. That is the thing to measure first if the track goes ahead.
