# cifar-messages

**Q.** Every graph result here rests on a synthetic task built around the mechanism under test. That
is the right way to ask whether a mechanism *can* work and says nothing about whether it does on data
nobody designed. On CIFAR-10, under one booleanization held fixed across every arm: does convolution
beat flat, and does one round of learned messages beat convolution?

**A. No, and no.**

| arm | width | accuracy | vs conv |
|---|---|---|---|
| flat FPTM *[baseline]* | 4096 | **0.3528** | +0.0429 |
| conv, no messages **[control]** | 264 | 0.3099 | — |
| conv + **dead** channel **[control]** | 296 | 0.3209 | +0.0109 |
| conv + random messages **[control]** | 296 | 0.3202 | +0.0103 |
| conv + learned messages | 296 | 0.3241 | +0.0141 |

**Convolution loses to flat by 4.3 points.** [`convolutional`](../convolutional/) found the opposite
on MNIST, +0.0216 at the same clause count. The Stage 3 result is MNIST-specific and was reported
without a second dataset behind it. A patch of CIFAR grayscale is presumably not a discriminative
unit the way an MNIST stroke fragment is, and max-over-patches throws away the global layout the
harder task needs — but that is a hypothesis, not something measured here.

**Learned messages contribute nothing measurable.** The +0.0141 looks like a result until the dead
channel is run: 32 bits wired permanently to zero reproduce +0.0109 of it. The residual is +0.0032,
inside the seed spread (learned 0.3222–0.3256, dead 0.3136–0.3254).

**Why a dead channel is not inert**, which is the whole reason that arm exists. A field that is
always zero still adds 32 literals a clause can satisfy for free through their negations. That
inflates the clause's literal count, which moves the `L` growth gate — and
[`budget-paths`](../budget-paths/) already measured that gate as worth 13 points when it moves.
"Messages helped" and "32 dead bits perturbed the literal budget" are not distinguishable without
this arm, and here the second explanation covers three quarters of the effect.

**The channel diagnostic agreed in advance.** Learned message bits are live on 0.001–0.003 of
node-clause pairs here, against 0.115–0.308 on the synthetic task in
[`learned-messages`](../learned-messages/). The channel collapsed; the accuracy gain arrived anyway,
which is what made the dead control necessary rather than optional.

**What this does not overturn.** [`learned-messages`](../learned-messages/) compared learned against
a *random* channel of identical width and won on 3/3 seeds, so width was controlled there. A dead
channel could not have explained that result either: the task provably cannot be solved without
neighbour information, so a silent channel is pinned near the no-message arm's 0.5082 by
construction. The difference is that CIFAR-10 does not *require* messages, which is what leaves room
for a budget artifact to masquerade as one.

**Caveats.** Absolute accuracy is far below the GraphTM paper's CIFAR-10 figures — their
convolutional baseline is in the 60s — because the booleanization here is grayscale with 4
thermometer thresholds and nothing else. No comparison to their +3.86 is drawn or implied. 20,000
train, 5,000 test, 40 clauses per class, one patch size, one stride, 3 seeds. A stronger
booleanization or a finer stride could change the convolution result; [`convolutional`](../convolutional/)
found stride alone flipping the sign on MNIST.
