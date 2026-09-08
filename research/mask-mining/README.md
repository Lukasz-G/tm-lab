# mask-mining

**Q.** Is a fuzzy clause recoverable as a rule?

**A.** On MNIST, as a **conjunctive core plus a tolerance tail** — not the "k rules plus a tail" that
was hypothesised. Median clause: 195 distinct satisfied masks, top-10 covering 42%, but 11 of 23.5
literals satisfied in >=95% of firings.

Extracted standalone, positive-clause cores carry the class signal in 92% of cases at median 7.37x
lift; negative cores only 59%.

**Does not generalise** — see `imdb-readability/`. The governing quantity is `LF` / included-literals:
21% on MNIST, 1.7% on IMDb, where the decomposition is vacuous.
