#!/bin/bash
#SBATCH --account=def-ispading
#SBATCH --time=01:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

pip install --no-index --upgrade pip
pip install pandas
python Co60BVMPtest.py