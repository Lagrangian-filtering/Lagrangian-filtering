#!/bin/bash
# Generate a config for each filter width W (in micro cells) on a common region and submit it.
# Run in the lagfilt environment, from filter_scripts. Usage: ./scan_w_submit.sh [W ...]
# L (observer box) = min(1.5 W, 12). Widths the window cannot support are skipped, not forced.
set -u
DATA=/scratch/ih3/METHOD_simulations/khi_window_576x576x288_8x8x3_seed1/
CROP="96 96 48"     # common crop
PAD=12              # common margin so every W uses the same meso grid
cd "$(dirname "$0")"
Ws=("$@"); [ ${#Ws[@]} -eq 0 ] && Ws=(2 4 6 8 10 12)
for W in "${Ws[@]}"; do
  L=$(python -c "print(min(1.5*$W, 12.0))")
  if python make_mhd_config.py "$DATA" --name "khi576_w$W" --crop-size $CROP --pad-cells $PAD \
        --filter-width "$W" --box-len "$L" --num-t-slices 1 --coarse 4; then
    sbatch "submit_khi576_w$W.sh"
  else
    echo "W=$W skipped: window check failed (see message above)"
  fi
done
