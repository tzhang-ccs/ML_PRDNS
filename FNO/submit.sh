#!/bin/bash
#SBATCH -q regular
##SBATCH -q debug
#SBATCH -A m4402
#SBATCH -J dns

#SBATCH  --nodes=1
#SBATCH  --output=debug.%j
#SBATCH  --error=error.%j
#SBATCH  --exclusive
#SBATCH  --time=12:00:00
#SBATCH  --constraint=gpu

ulimit -s unlimited
ulimit -n 1000000

python FNO_2d_bigdata.py -p Train
