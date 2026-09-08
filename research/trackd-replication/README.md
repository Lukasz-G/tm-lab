# trackd-replication

**Q.** Do the Track D conclusions survive other datasets?

**A.** Direction yes, magnitude no. **All five variants lose on MNIST, Fashion-MNIST and CIFAR-10, on
every seed — 15 of 15.**

| variant | MNIST | Fashion | CIFAR |
|---|---|---|---|
| proportional feedback | -0.0042 | -0.0085 | -0.0228 |
| proportional-idle | -0.0040 | -0.0064 | -0.0204 |
| confidence-weighted cost | -0.0028 | -0.0019 | -0.0060 |
| anneal `LF` | -0.0050 | -0.0039 | -0.0141 |
| `L` gates Type II | -0.1339 | -0.0615 | -0.0831 |

Damage scales with task difficulty (baselines 0.97, 0.86, 0.33), so the MNIST numbers were the most
flattering case. Effect sizes are not properties of FPTM; only the sign is.
