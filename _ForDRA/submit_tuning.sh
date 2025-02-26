#!/bin/bash
#SBATCH --account=def-ispading
#SBATCH --time=6-00
#SBATCH --mail-user=cbnorthway@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
module load python/3.10
module load python scipy-stack
source ENV/bin/activate
pip install --no-index --upgrade pip
pip install numba --no-index
srun python BVOptimization_Alliance.py
