#!/usr/bin/env bash
# Run the incremental-VI experiments and write ALL results to the data folder
# ($OUT, default /data). Nothing is written outside $OUT.
#
#   docker run --rm -v /ABS/PATH/TO/DATA:/data hot-start-vi bash run_all.sh
set -euo pipefail

OUT="${OUT:-/data}"
mkdir -p "$OUT"
TS="$(date +%Y%m%d-%H%M%S)"
echo "writing results to $OUT  (run $TS)"

# 1. COI Algorithms 1 & 2 on the PRISM/QVBS MDP benchmarks (EXACT values)
EXACT=1 NUM="${NUM:-40}" python -u run_prism_mdp.py 2>&1 \
    | tee "$OUT/coi_prism_${TS}.txt"

# 2. Undershooting vs VI (iteration counts) on consensus
python -u run_undershoot_prism.py consensus-coin2-K2 consensus-coin2-K4 --num 20 2>&1 \
    | tee "$OUT/undershoot_${TS}.txt"

# 3. Per-removal evaluation table (Overview spec) for a couple of models
python -u coi_table.py models/archive/cav23-saynt/refuel-06 --num 8 2>&1 \
    | tee "$OUT/coi_table_refuel06_${TS}.txt"

# 4. Bachelor-thesis IDAR reproduction on drone-4-1 (k=1)
python -u reproduce_eidar.py models/archive/cav23-saynt/drone-4-1 600 2>&1 \
    | tee "$OUT/eidar_drone41_${TS}.txt"

echo "DONE -> results in $OUT"
ls -la "$OUT"
