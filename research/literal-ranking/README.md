# literal-ranking

**Q.** Use the confidence gradient to rank a clause's 3,800 literals and show the top few?

**A.** No — the gradient **saturates** on IMDb. Included automata sit at median 255 with 99.9% at the
ceiling, because these clauses fire on 78% of documents and reinforcement pins them at max. MNIST
clauses fire on ~9%, which is why a spread appeared there.

So reset eviction yields a usable gradient only for infrequently-firing clauses.

The frequency baseline is no better (every truncation at P~0.50). Both signals measure *how often a
literal holds*, not *how much it separates classes* — which is the real flaw.
