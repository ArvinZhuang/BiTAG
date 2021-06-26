#!/bin/bash
#SBATCH -N 4
#SBATCH --job-name=absTtitle
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=50G
#SBATCH -o print_train.txt
#SBATCH -e error_train.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --cpus-per-task=1

module load anaconda/3.6
source activate /scratch/itee/uqszhuan/absTtitle/env
module load cuda/10.0.130
module load gnu/5.4.0
module load mvapich2

srun python3 train.py