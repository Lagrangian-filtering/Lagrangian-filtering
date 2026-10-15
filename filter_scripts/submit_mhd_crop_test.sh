#!/bin/bash
#SBATCH --job-name=lagfilt-crop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # matches n_cpus in the config
#SBATCH --time=01:00:00
#SBATCH --partition=<partition>
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load conda                  # use the module name you used to create the env
eval "$(conda shell.bash hook)"
conda activate lagfilt

export OMP_NUM_THREADS=1           # one BLAS thread per multiprocessing worker
export MPLBACKEND=Agg

mkdir -p /scratch/ih3/METHOD_simulations/khi_window_384x384x192_8x8x3_seed1/pickled
cd ~/Lagrangian-filtering/filter_scripts   # the driver uses ../master_files relative to here
python pickling_meso_mhd.py config_mhd_crop_test.txt
