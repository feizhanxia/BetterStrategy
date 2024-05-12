#!/bin/bash
#SBATCH -p TH_HPC3N
#SBATCH -J train10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 10:00:00
#SBATCH -o output/slurm_output/train10.out

module purge
module load loginnode
source activate py3.10

python train10.py