#!/bin/bash
#SBATCH --account=k1124
#SBATCH --job-name=qdwhpartial
#SBATCH --output=qdwhpartial-%j.out
#SBATCH --error=qdwhpartial-%j.err
#SBATCH --nodes=12
#SBATCH --time=00:02:00

export CRAYPE_LINK_TYPE=dynamic
export MKL_NUM_THREADS=1

srun --ntasks=384 --hint=nomultithread --ntasks-per-node=32 --ntasks-per-socket=16  ./main --nprow 12 --npcol 32 --b 64 --cond 1.e16 --niter 1 --n_range 10240:10240:10240   --check


echo "== Node lists:", $SLURM_JOB_NODELIST
