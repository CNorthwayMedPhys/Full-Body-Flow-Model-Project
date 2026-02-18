#!/bin/bash
#SBATCH --account=def-ispading
#SBATCH --time=167:00:00
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=cbnorthway@gmail.com

pip install --no-index --upgrade pip
pip install pandas
python Co60BVMP.py