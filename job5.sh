#!/bin/bash
#SBATCH -p TH_HPC3N
#SBATCH -J train11
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 10:00:00
#SBATCH -o output/slurm_output/train11.out

module purge
module load loginnode
source activate py3.10

python train11.py