# imdb-readability

**Q.** Do extracted cores read as rules when literals are words?

**A.** No. Reproduces the published IMDb number (0.9012 vs 0.9015), then finds cores of 3,375-4,127
literals: precise (0.979) and unreadable.

The reason is arithmetic. The clause holds 3,843 literals with `LF`=64, so it tolerates **1.7%**
failing and is already a near-strict conjunction; the 95% core threshold then selects 98% of it.
MNIST ran at 21%. **`LF` / included-literals is the diagnostic**, usable in advance.

Unplanned: **not one core requires a term to be present.** The model is a pure blacklist. Also the
most extreme `L`-as-growth-gate case seen — `L`=64 with clauses at ~3,800 literals, 58x over.
