#!/bin/bash
# Submit a grid of abutting crops whose meso grids tile one large region (W=4, L=6, one time slice).
# Run in the lagfilt environment, from filter_scripts.
# Usage: [X0=144] [Y0=144] [NX=4] [NY=4] [Z0=120] ./mosaic_submit.sh
# Crops are 96x96x48 cells, stepped by 72 cells so the meso interiors (96 - 2*12 pad) abut exactly
# and share one lattice (72 and 12 are multiples of the coarse factor 4). X0, Y0, Z0 are the first
# crop's start cells; the mosaic covers about 72*NX x 72*NY cells in x, y. Default: 4x4 = 16 jobs.
set -u
DATA=/scratch/ih3/METHOD_simulations/khi_window_576x576x288_8x8x3_seed1/
X0=${X0:-144}; Y0=${Y0:-144}; NX=${NX:-4}; NY=${NY:-4}; Z0=${Z0:-120}
STEP=72
cd "$(dirname "$0")"
for ((a=0; a<NX; a++)); do
  for ((b=0; b<NY; b++)); do
    SX=$((X0 + STEP*a)); SY=$((Y0 + STEP*b))
    if [ $SX -lt 0 ] || [ $SX -gt 480 ] || [ $SY -lt 0 ] || [ $SY -gt 480 ]; then
      echo "crop ($SX,$SY,$Z0) outside the 576x576 box (valid starts 0..480): skipped"; continue
    fi
    NAME="khi576_m_${SX}_${SY}_${Z0}"
    if python make_mhd_config.py "$DATA" --name "$NAME" --crop-size 96 96 48 --crop-start $SX $SY $Z0 \
          --pad-cells 12 --filter-width 4 --box-len 6 --num-t-slices 1 --coarse 4; then
      sbatch "submit_${NAME}.sh"
    else
      echo "crop ($SX,$SY,$Z0) refused by the generator"
    fi
  done
done
