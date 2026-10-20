#!/bin/bash
# The data directory defaults to the 576x576x288 run; set KHI_DATA=/path/to/run (no trailing slash) to use another.
# Generate and submit one filtering job per crop region at fixed filter width.
# Run in the lagfilt environment, from filter_scripts. Usage: ./scan_regions_submit.sh [i,j,k ...]
# Each argument is the first cell of a 96x96x48 crop, e.g. 40,100,120. Default: 8 spread-out regions.
set -u
DATA="${KHI_DATA:-/scratch/ih3/METHOD_simulations/khi_window_576x576x288_8x8x3_seed1}/"
W=4; L=6; CROP="96 96 48"; PAD=12
cd "$(dirname "$0")"
Rs=("$@")
[ ${#Rs[@]} -eq 0 ] && Rs=(40,100,120 240,100,120 440,100,120 40,380,120 240,380,120 440,380,120 240,240,48 240,240,192)
for R in "${Rs[@]}"; do
  IFS=, read -r I J K <<< "$R"
  NAME="khi576_r${I}_${J}_${K}"
  if python make_mhd_config.py "$DATA" --name "$NAME" --crop-size $CROP --crop-start $I $J $K --pad-cells $PAD \
        --filter-width $W --box-len $L --num-t-slices 1 --coarse 4; then
    sbatch "submit_${NAME}.sh"
  else
    echo "region $R skipped (generator refused; see message above)"
  fi
done
