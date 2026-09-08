# annealed-lf

**Q.** Anneal `LF` fuzzy-early, strict-late?

**A.** No, and the schedule is irrelevant — **only the endpoint matters.** Both arms ending at `LF`=1
finish near 0.94; both ending at `LF`=5 finish at 0.971, whatever their start.

Best-versus-final exposes the damage: down-annealed models peak at 0.967 early and degrade to 0.937
as `LF` falls. Quoting best accuracy alone understates it eightfold.

Two controls earned their place: rescaling `T` recovers half the final loss and none of the peak
loss, and a reversed strict->fuzzy arm ties the baseline.
