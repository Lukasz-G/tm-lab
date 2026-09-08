#!/usr/bin/env bash
# Launch the sweep in parallel, one single-threaded Julia process per point.
#
# T follows the FPTM paper's multiclass relation T ~ sqrt(CLAUSES/2 * LF) with LF = 5, except for
# the two 250-clause points at T = 10, which hold T at its 40-clause value to check the relation is
# load-bearing rather than decorative.
#
# Points already present in out/ are skipped, so this is re-runnable and fills gaps.
# Requires the bit cache: julia --project=. research/clause-count/prep.jl
cd "$(dirname "$0")/../.."
EPOCHS=${1:-25}

run_point() {   # clauses T seed
  f=$(printf "research/clause-count/out/c%04d_T%02d_s%d.txt" "$1" "$2" "$3")
  if [ -f "$f" ]; then echo "skip $f"; return; fi
  julia --project=. research/clause-count/run.jl "$1" "$2" "$3" "$EPOCHS" &
}

for SEED in 20260908 11; do
  run_point   40 10 $SEED
  run_point  100 16 $SEED
  run_point  250 25 $SEED
  run_point  500 35 $SEED
  run_point 1000 50 $SEED
  run_point  250 10 $SEED
done
wait
echo "sweep complete"
