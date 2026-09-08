# pileup-verify

**Q.** Is the automaton pile-up real in Hnilov's own code, and is `L` the cause?

**A.** Yes to both.

| implementation | `L` | accuracy | included state min/med/max | above threshold |
|---|---|---|---|---|
| FuzzyPatternTM | 10 | 0.9536 | 128 / **128** / 133 | **8.1%** |
| TMCore | 10 | 0.9517 | 128 / **128** / 136 | **7.7%** |
| FuzzyPatternTM | 100000 | 0.3114 | 128 / 152 / 255 | **97.0%** |
| TMCore | 100000 | 0.3422 | 128 / 158 / 255 | **97.0%** |

So it is a property of FPTM as published, not of this reimplementation — and the two agree to within
a percentage point, cross-validating TMCore on something no test covered.

Removing `L` is not an option: 0.95 -> 0.31, clauses at ~530 literals.

Scope: `L` is not in Granmo's 2018 TM. It is a later addition.
