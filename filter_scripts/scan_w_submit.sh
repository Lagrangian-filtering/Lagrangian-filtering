#!/bin/bash
# Generate a config for each filter width W (in micro cells) on a common region and submit it.
# Run in the lagfilt environment, from filter_scripts. Usage: [REGION=i,j,k] ./scan_w_submit.sh [W ...]
# REGION is the first cell of the 96x96x48 crop; unset = centred crop (names khi576_w<W>).
# With REGION=240,380,120 the names are khi576_r240_380_120_w<W>.
# L (observer box) = min(1.5 W, 12). Widths the window cannot support are skipped, not forced.
set -u
DATA=/scratch/ih3/METHOD_simulations/khi_window_576x576x288_8x8x3_seed1/
CROP="96 96 48"     # common crop
PAD=12              # common margin so every W uses the same meso grid
cd "$(dirname "$0")"
REGION=${REGION:-}
if [ -n "$REGION" ]; then
  IFS=, read -r I J K <<< "$REGION"
  TAG="r${I}_${J}_${K}_"; START="--crop-start $I $J $K"
else
  TAG=""; START=""
fi
Ws=("$@"); [ ${#Ws[@]} -eq 0 ] && Ws=(2 4 6 8 10 12)
for W in "${Ws[@]}"; do
  L=$(python -c "print(min(1.5*$W, 12.0))")
  NAME="khi576_${TAG}w$W"
  if python make_mhd_config.py "$DATA" --name "$NAME" --crop-size $CROP $START --pad-cells $PAD \
        --filter-width "$W" --box-len "$L" --num-t-slices 1 --coarse 4; then
    sbatch "submit_${NAME}.sh"
  else
    echo "W=$W skipped: window check failed (see message above)"
  fi
done
