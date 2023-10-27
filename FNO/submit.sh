#!/bin/bash
#SBATCH -q regular
##SBATCH -q debug
#SBATCH -A m4402
#SBATCH -J dns

#SBATCH  --nodes=1
#SBATCH  --output=/pscratch/sd/z/zhangtao/FNO_PR_DNS/logs/debug.%j
#SBATCH  --error=/pscratch/sd/z/zhangtao/FNO_PR_DNS/logs/error.%j
#SBATCH  --exclusive
#SBATCH  --time=7:00:00
#SBATCH  --constraint=gpu

ulimit -s unlimited

python FNO_2d_bigdata.py  -p Train -r vel
