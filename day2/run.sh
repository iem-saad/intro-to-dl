#!/bin/bash
#SBATCH --account=project_462000698
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --reservation=PDL_CSC_GPU

module purge
module use /appl/local/csc/modulefiles/
module load pytorch

COURSE_SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"

export DATADIR=$COURSE_SCRATCH/data
export TORCH_HOME=$COURSE_SCRATCH/torch-cache
export HF_HOME=$COURSE_SCRATCH/hf-cache
export MLFLOW_TRACKING_URI=$COURSE_SCRATCH/data/users/$USER/mlruns

set -xv
python3 $*
