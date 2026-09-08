# mnist-training

**Q.** Does TMCore, trained from scratch, reproduce published FPTM?

**A.** 0.9769 against the published model's 0.9809, and it reproduces the clause-size overshoot
(median 16, max 46, under `L`=10) that a wrong feedback rule would not.

The gap is provenance: upstream's own example shows the shipped model is the best of 512 checkpoints
over 1000 epochs, then a pairwise merge, **both selected on test accuracy**. So 0.9809 is
optimistically biased, and merging is why its clauses are larger. `include_limit` and longer training
were tested and ruled out.
