#!/bin/bash
# The data directory defaults to the 576x576x288 run; set KHI_DATA=/path/to/run (no trailing slash) to use another.
# Analyse the pickles from scan_w_submit.sh, then tabulate against W.
# Run in the lagfilt environment, from filter_scripts, once the jobs have finished.
# Usage: [REGION=i,j,k] [NSLICES=n] [COARSE=c] [MODE=ohms|gradient] ./scan_w_analyse.sh [W ...]
#   REGION, NSLICES, COARSE must match the submit step.
#   MODE=ohms (default): regional regression of the resistive/dynamo/Hall terms -> scan_results[...]/
#   MODE=gradient: the second-order (gradient) model of E_tilde (needs NSLICES=3) -> gradient_scan[...]/
set -u
PICK="${KHI_DATA:-/scratch/ih3/METHOD_simulations/khi_window_576x576x288_8x8x3_seed1}/pickled"
cd "$(dirname "$0")"
REGION=${REGION:-}
NSLICES=${NSLICES:-1}
COARSE=${COARSE:-4}
MODE=${MODE:-ohms}
if [ -n "$REGION" ]; then
  IFS=, read -r I J K <<< "$REGION"
  TAG="r${I}_${J}_${K}_"; RTAG="_r${I}_${J}_${K}"
else
  TAG=""; RTAG=""
fi
SUFFIX=""
[ "$NSLICES" != "1" ] && SUFFIX="${SUFFIX}_${NSLICES}sl"
[ "$COARSE" != "4" ] && SUFFIX="${SUFFIX}_c${COARSE}"
# tile size in meso cells scales with the meso spacing so tiles cover the same physical area
BX=$((24 / COARSE)); BZ=$((12 / COARSE))
if [ "$MODE" = "gradient" ]; then
  OUT="gradient_scan${RTAG}${SUFFIX}"
else
  OUT="scan_results${RTAG}${SUFFIX}"
fi
Ws=("$@"); [ ${#Ws[@]} -eq 0 ] && Ws=(2 4 6 8 10 12)
mkdir -p "$OUT"
for W in "${Ws[@]}"; do
  f="$PICK/resMHD3d_khi576_${TAG}w${W}${SUFFIX}.pickle"
  if [ ! -f "$f" ]; then echo "W=$W: no pickle ($f)"; continue; fi
  if [ "$MODE" = "gradient" ]; then
    python regress_gradient_model.py "$f" --block $BX $BX $BZ --min-points 100 --label "$W" \
        --outdir "$OUT/w$W" > "$OUT/w$W.log" 2>&1 || echo "W=$W: regression failed, see $OUT/w$W.log"
  else
    python regress_ohms_law.py "$f" --block $BX $BX $BZ --min-points 100 --label "$W" \
        --outdir "$OUT/w$W" > "$OUT/w$W.log" 2>&1 || echo "W=$W: regression failed, see $OUT/w$W.log"
  fi
done
if [ "$MODE" = "gradient" ]; then
  python summarize_gradient_scan.py "$OUT"
else
  python summarize_w_scan.py "$OUT"
fi
