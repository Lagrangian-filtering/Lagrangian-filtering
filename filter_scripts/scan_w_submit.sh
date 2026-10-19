#!/bin/bash
# Generate a config for each filter width W (in micro cells) on a common region and submit it.
# Run in the lagfilt environment, from filter_scripts.
# Usage: [REGION=i,j,k] [NSLICES=n] [COARSE=c] ./scan_w_submit.sh [W ...]
#   REGION  first cell of the 96x96x48 crop; unset = centred crop
#   NSLICES number of meso time slices (default 1; the gradient model needs 3)
#   COARSE  meso spacing in micro cells (default 4; 2 resolves gradients better but has ~8x the points)
# Names: khi576_[r<i>_<j>_<k>_]w<W>[_<n>sl][_c<COARSE>]; the suffixes appear only for non-default values,
# so NSLICES=1 COARSE=4 reproduces the original names, and NSLICES=3 COARSE=2 gives khi576_w4_3sl_c2.
# L (observer box) = min(1.5 W, 12). Widths the window cannot support are skipped, not forced.
set -u
DATA=/scratch/ih3/METHOD_simulations/khi_window_576x576x288_8x8x3_seed1/
CROP="96 96 48"     # common crop
PAD=12              # common margin so every W uses the same meso grid
cd "$(dirname "$0")"
REGION=${REGION:-}
NSLICES=${NSLICES:-1}
COARSE=${COARSE:-4}
if [ -n "$REGION" ]; then
  IFS=, read -r I J K <<< "$REGION"
  TAG="r${I}_${J}_${K}_"; START="--crop-start $I $J $K"
else
  TAG=""; START=""
fi
SUFFIX=""
[ "$NSLICES" != "1" ] && SUFFIX="${SUFFIX}_${NSLICES}sl"
[ "$COARSE" != "4" ] && SUFFIX="${SUFFIX}_c${COARSE}"
Ws=("$@"); [ ${#Ws[@]} -eq 0 ] && Ws=(2 4 6 8 10 12)
for W in "${Ws[@]}"; do
  L=$(python -c "print(min(1.5*$W, 12.0))")
  NAME="khi576_${TAG}w${W}${SUFFIX}"
  if python make_mhd_config.py "$DATA" --name "$NAME" --crop-size $CROP $START --pad-cells $PAD \
        --filter-width "$W" --box-len "$L" --num-t-slices "$NSLICES" --coarse "$COARSE"; then
    sbatch "submit_${NAME}.sh"
  else
    echo "W=$W skipped: window check failed (see message above)"
  fi
done
