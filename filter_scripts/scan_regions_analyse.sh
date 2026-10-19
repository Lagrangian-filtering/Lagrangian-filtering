#!/bin/bash
# Regress each region's pickle, then tabulate. Run once the jobs have finished.
# Usage: ./scan_regions_analyse.sh [i,j,k ...]   (same defaults as scan_regions_submit.sh)
set -u
PICK=/scratch/ih3/METHOD_simulations/khi_window_576x576x288_8x8x3_seed1/pickled
cd "$(dirname "$0")"
Rs=("$@")
[ ${#Rs[@]} -eq 0 ] && Rs=(40,100,120 240,100,120 440,100,120 40,380,120 240,380,120 440,380,120 240,240,48 240,240,192)
mkdir -p region_results
for R in "${Rs[@]}"; do
  IFS=, read -r I J K <<< "$R"
  NAME="khi576_r${I}_${J}_${K}"
  f="$PICK/resMHD3d_${NAME}.pickle"
  if [ ! -f "$f" ]; then echo "region $R: no pickle yet"; continue; fi
  python regress_ohms_law.py "$f" --block 6 6 3 --min-points 100 --label "$R" \
      --outdir "region_results/$NAME" > "region_results/$NAME.log" 2>&1 || echo "region $R: regression failed, see region_results/$NAME.log"
done
python summarize_regions.py region_results
