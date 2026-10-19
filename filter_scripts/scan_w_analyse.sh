#!/bin/bash
# Run the regional regression on each pickle from scan_w_submit.sh, then tabulate vs W.
# Run in the lagfilt environment, from filter_scripts, once the jobs have finished.
# Usage: [REGION=i,j,k] ./scan_w_analyse.sh [W ...]   (REGION must match the submit step)
# Results: scan_results/ (centred crop) or scan_results_r<i>_<j>_<k>/ (with REGION).
set -u
PICK=/scratch/ih3/METHOD_simulations/khi_window_576x576x288_8x8x3_seed1/pickled
cd "$(dirname "$0")"
REGION=${REGION:-}
if [ -n "$REGION" ]; then
  IFS=, read -r I J K <<< "$REGION"
  TAG="r${I}_${J}_${K}_"; OUT="scan_results_r${I}_${J}_${K}"
else
  TAG=""; OUT="scan_results"
fi
Ws=("$@"); [ ${#Ws[@]} -eq 0 ] && Ws=(2 4 6 8 10 12)
mkdir -p "$OUT"
for W in "${Ws[@]}"; do
  f="$PICK/resMHD3d_khi576_${TAG}w$W.pickle"
  if [ ! -f "$f" ]; then echo "W=$W: no pickle ($f)"; continue; fi
  python regress_ohms_law.py "$f" --block 6 6 3 --min-points 100 --label "$W" \
      --outdir "$OUT/w$W" > "$OUT/w$W.log" 2>&1 || echo "W=$W: regression failed, see $OUT/w$W.log"
done
python summarize_w_scan.py "$OUT"
