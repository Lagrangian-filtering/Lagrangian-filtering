#!/bin/bash
# Run the regional regression on each pickle from scan_w_submit.sh, then tabulate vs W.
# Run in the lagfilt environment, from filter_scripts, once the jobs have finished.
set -u
PICK=/scratch/ih3/METHOD_simulations/khi_window_576x576x288_8x8x3_seed1/pickled
cd "$(dirname "$0")"
Ws=("$@"); [ ${#Ws[@]} -eq 0 ] && Ws=(2 4 6 8 10 12)
mkdir -p scan_results
for W in "${Ws[@]}"; do
  f="$PICK/resMHD3d_khi576_w$W.pickle"
  if [ ! -f "$f" ]; then echo "W=$W: no pickle ($f)"; continue; fi
  python regress_ohms_law.py "$f" --block 6 6 3 --min-points 100 --label "$W" \
      --outdir "scan_results/w$W" > "scan_results/w$W.log" 2>&1 || echo "W=$W: regression failed, see scan_results/w$W.log"
done
python summarize_w_scan.py scan_results
