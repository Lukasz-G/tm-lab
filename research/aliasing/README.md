# aliasing

**Q.** Over a sparse distributed code, is tolerating a missing *bit* still a partial match, or a
corrupted symbol?

**A.** `hypervector_bits` is the lever, not match granularity. Expected number of other symbols that
alone drive a clause to within one of a full vote, at V=100: 99 at H=1, 12.2 at H=2/D=32, 5.7e-4 at
H=4/D=256. Danger zone is H<=2 and ends abruptly.

Symbol-granular fuzziness is the *coarser* option, not the safer one — forgiving a whole symbol
turns a 3-symbol clause into a 2-symbol one.

Untested: message symbols, whose codes are cyclic shifts rather than independent draws.
