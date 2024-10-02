#!/bin/bash
####################################
# JOB INFO: parallel, single node
####################################

#SBATCH --output=serial%A.out
#SBATCH --nodes=1      
#SBATCH --ntasks=1                
#SBATCH --time=00:10:00 

#SBATCH --mail-type=begin        
#SBATCH --mail-type=fail 
#SBATCH --mail-type=end       
#SBATCH --mail-user=celora@ice.csic.es


####################################
# LOADING THE CONDA ENVIRONMENT 
####################################

module load conda/py3-latest
# I get warnings to use 'conda deactivate' instead but that doesn't work
source deactivate 
conda activate myenv


####################################
# LAUNCHING THE JOBS
####################################

python -u visualize_3Dslices.py config_visualize.txt

# python -u test_meso3D.py config_mesomodel.txt

