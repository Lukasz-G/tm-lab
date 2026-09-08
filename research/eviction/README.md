# eviction

**Q.** Can a clause have a size limit *and* a confidence gradient?

**A.** Yes, via **reset eviction** — an eroded included literal drops straight below the threshold, so
eviction costs one hit whatever the confidence.

| arm | MNIST | Fashion | CIFAR | above threshold (MNIST) |
|---|---|---|---|---|
| reference | 0.9713 | 0.8584 | 0.3313 | 6.7% |
| naive unfreeze | -0.1668 | -0.1123 | -0.1708 | 98.2% |
| **+ reset** | **+0.0003** | **-0.0012** | **-0.0040** | **94.0%** |
| + proportional | -0.0204 | -0.0198 | -0.0517 | 98.5% |

Not free: MNIST is a tie, Fashion and CIFAR lose small amounts outside seed noise. But 0-0.4 points
against 16.9 for the naive version.

Worth nothing on its own — a gradient only matters if something reads it. See `confidence-payoff/`.
