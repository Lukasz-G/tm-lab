# booleanization

**Q.** How much accuracy lives in the encoder rather than the machine?

**A.** On MNIST, +0.0021 for four bits per pixel instead of one — and the confound is larger than the
effect. `s` is derived as `width/S`, so widening 784 -> 3136 bits quadruples the forgetting rate
unless `S` scales:

```
encoder alone (s held)      +0.0021
forgetting rate (s 6 -> 25) +0.0025
naive 1-bit vs 4-bit test   +0.0046
```

Also: fitted thresholds do **not** beat upstream's hardcoded quartiles, and a 2-bit thermometer beats
every 4-bit arm.

The convolutional-kernel claim that motivated this track is a much stronger intervention and remains
untested.
