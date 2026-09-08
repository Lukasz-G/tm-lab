# vote-histogram

**Q.** Is FPTM's fuzziness load-bearing or decorative?

**A.** Load-bearing. On the published 40-clause MNIST model, of the 16.3% of evaluations that vote at
all, **91.6% are strictly interior** and only 8.4% reach the ceiling. Not bimodal — monotonic decay.
Per-clause interior fraction median 0.936.

Literal satisfaction spread (median 0.638) says the satisfied mask has structure.
