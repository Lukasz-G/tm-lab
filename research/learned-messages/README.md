# learned-messages

**Q.** Stage 4 used *identity* messages — a node receives its neighbours' features — which is a
3-gram window with a graph label on it. Do **learned** messages, GraphTM's actual mechanism, beat a
fixed channel?

**Task.** Stage 4's cannot test this: identity messages already score 1.0000, leaving no headroom.
The question only has teeth when the channel is too narrow to carry the raw features. So: alphabet
of 32, length 12, label = a symbol from set A immediately followed by one from set B. The useful
message from a right neighbour is **one bit** ("am I in B?"); carrying it as identity costs 32.

**A.** Learned messages match the full-width ceiling at 25% less node width, and beat a random
channel.

| arm | node width | accuracy |
|---|---|---|
| no messages **[control]** | 32 | **0.5082** |
| identity messages *[ceiling]* | 96 | 1.0000 |
| random messages **[control]** | 72 | 0.9523 |
| **learned messages** | 72 | **1.0000** (3/3 seeds) |

The random-channel control is the one that matters: without it, "learned messages work" would be
indistinguishable from "any 40-bit channel works". Random reaches 0.9523 and varies across seeds
(0.923–0.998); learned hits 1.0000 on every seed.

The channel diagnostic agrees. Message bits fire on 5–6% of node-clause pairs under the random
channel and 11–31% under the learned one, so learning made the channel carry more, not just
different.

**Mechanism.** One clause bank of width `K + 2·nclauses`, evaluated twice. At round 0 the message
inputs are zero, so a clause's round-0 output is its response to raw features; those outputs become
round-1 messages. The bank is trained by round-1 feedback and its round-0 behaviour shifts as a side
effect — the same automata at both depths, which is how GraphTM trains depth. This is a reading of
that mechanism, not a port.

**Watch out.** At 4 epochs learned scored 0.5325, below random, which looked like the classifier
learning to depend on message literals and thereby silencing the channel. It was slow convergence,
not collapse — by 6 epochs it led. The diagnostic was added to tell those apart rather than guess.

**Caveats.** One synthetic task, built so the message is compressible to a single bit; random already
reaches 0.95, so the margin is 0.048. One configuration, 3 seeds. Nothing here says learned messages
help on a real problem.
