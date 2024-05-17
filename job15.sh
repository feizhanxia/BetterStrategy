#!/bin/bash
#SBATCH -p TH_HPC3N
#SBATCH -J train15
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 5:00:00
#SBATCH -o output/slurm_output/train15.out

module purge
module load loginnode
source activate py3.10

python train15.py