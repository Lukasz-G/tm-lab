# clause-count

**Calibration, not a finding.** That more clauses help, and that `T` must scale with them by
`T ≈ sqrt(CLAUSES/2 × LF)`, is stated in the FPTM paper. This sweep re-measured it, which was not
worth the compute it cost. It is kept for one reason only: it shows the stack reaches published-range
accuracy, which is evidence about *this implementation* rather than about Tsetlin machines.

CIFAR-10, best encoder from [`booleanization-cifar`](../booleanization-cifar/) (RGB ×2 thermometer +
fixed edge kernels, 10194 bits), `s` held at 33, 20,000 train, 5,000 test, 25 epochs, 2 seeds.

| clauses/class | `T` | accuracy |
|---|---|---|
| 40 | 10 | 0.5388 |
| 100 | 16 | 0.5847 |
| 250 | 25 | 0.6120 |
| 500 | 35 | 0.6210 |
| 1000 | 50 | 0.6324 *(1 seed)* |
| *250* | *10 — `T` not scaled* | *0.5645* |

**0.6324 lands in the published range**, so the remaining gap after
[`booleanization-cifar`](../booleanization-cifar/) was clause count, not a defect in the evaluator.
Together with the encoder, 0.3528 → 0.6324 is accounted for. Most points peaked on the final epoch,
so the column is a lower bound at every row.

The `T`=10 control at 250 clauses lands *below* 100 clauses at `T`=16, i.e. raising clause count
without raising `T` goes backwards. Also published, also not news.

**Reproduce:** `julia --project=. research/clause-count/prep.jl` caches the encoded bits once, then
`bash research/clause-count/sweep.sh` runs one process per point and skips points already in `out/`.
Encoding inside each process instead cost 600 MB per process and killed 7 of 12 on memory.

**The lesson worth keeping** is about method, not about TMs: state what result would surprise someone
who has read the papers *before* running anything. There was no such result here, and one cheap point
would have served the calibration purpose that two CPU-hours were spent on.
