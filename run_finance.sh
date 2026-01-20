#!/bin/bash
#SBATCH --job-name=mixed_fbm
#SBATCH --output=mixed_fbm_%j.out
#SBATCH --error=mixed_fbm_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G

cd /sci/labs/orzuk/orzuk/github/BandedDecomposition
source ~/.bashrc
conda activate pymol-env

python finance_example.py --model mixed_fbm --n 500 --strategy both --hres 0.01 --hmin 0.5 --parallel --workers 4
