# booleanization-cifar

**Q.** [`cifar-messages`](../cifar-messages/) put flat FPTM at 0.3528 on CIFAR-10 under grayscale
with four thermometer thresholds, against published convolutional-TM figures in the 60s. If a large
share of reported TM accuracy lives in booleanization rather than in the machine — the premise of
this track, and something FPTM's own Fashion-MNIST result leans on by using fixed convolutional
kernels its paper disclaims as not part of the TM — then changing only the encoder should move that
number a long way.

**A. +18.5 points, with the machine untouched.**

| encoder | width | accuracy | vs baseline |
|---|---|---|---|
| grayscale ×4, no kernels *[baseline]* | 4096 | 0.3528 | — |
| grayscale ×6, no kernels **[width control]** | 6144 | 0.3620 | +0.0092 |
| RGB ×2, no kernels | 6144 | 0.4295 | +0.0767 |
| grayscale ×4 + edge kernels | 5446 | 0.4937 | +0.1409 |
| **RGB ×2 + edge kernels** | 10194 | **0.5381** | **+0.1853** |

Flat FPTM in every arm — no patches, no messages, same 40 clauses per class, same `T`, `L`, `LF`.
Only the encoder moves.

**The width control is what makes the colour row readable.** RGB at 2 thresholds differs from the
baseline in colour *and* per-pixel resolution *and* width, so on its own it measures three things.
Grayscale at 6 thresholds is exactly 6144 bits — the same width, the same `s` — and gains only
**+0.0092**. So width buys essentially nothing here and colour is worth **+0.0675** against its
proper control.

**`s` is held constant in every arm.** `s = round(width / S)` is the number of literals Type Ib
erodes per clause, so it scales with input width; [`booleanization`](../booleanization/) measured that
confound as larger than the encoder effect it was meant to be studying. Every arm sets `S = width/33`
so a wider encoder does not silently get weaker forgetting.

**The two factors are mostly complementary.** Colour is +0.0675 (against the width control), kernels
+0.1409, and together +0.1853 against an additive +0.2084 — so they overlap by about a tenth. Edge
structure and colour are largely different information, which is what makes stacking them worthwhile.

**Scale.** The best *algorithmic* result anywhere in this repo is +0.0216, on MNIST, and it reversed
on CIFAR-10. The encoder is worth roughly eight times that, in the same place the algorithm work
failed. That ordering is the finding, more than the absolute number is.

**Caveats, and the gap that remains.** 0.5381 is still short of published CIFAR-10 figures in the
60s. Two obvious reasons are unaddressed here: 40 clauses per class is very small for this dataset,
and published convolutional TMs put the convolution *inside* the machine rather than only in the
encoder. So this closes roughly two thirds of the gap it set out to explain, not all of it. One
dataset for the colour factor, three for the kernels
([`kernel-replication`](../kernel-replication/)), one patch size, one pooling factor, 3 seeds.
