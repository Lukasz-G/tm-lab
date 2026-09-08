# format-crosscheck

**Q.** Is the TMCore model format actually readable by a second implementation?

**A.** Yes. A model trained in Julia is read by `tools/read_tmcore.py` — pure standard library,
written against the spec rather than the writer — which reproduces **every per-class score exactly**
on 50 MNIST cases, for both miss-cost policies.

Exercises chunk padding at width 784, array ordering, the policy codes and the stored-count check.
Inference-only models are 8x smaller and predict identically.
