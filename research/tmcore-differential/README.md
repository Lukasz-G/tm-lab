# tmcore-differential

**Q.** Does TMCore's bit-packed evaluator match FuzzyPatternTM's index-list one?

**A.** Exactly. **4,000,000 comparisons on the published model, 0 mismatches.** The satisfied mask
decodes to a naive recomputation with 0 disagreements, and import round-trips losslessly.

`FlatLF` diverges on exactly 20,000 comparisons — 2 clause slots x 10,000 inputs, the band
`ceiling-divergence` predicted — confirming the ceiling policy reaches the evaluator.
