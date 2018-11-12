#!/bin/bash

#SBATCH --job-name=testbatch 
#SBATCH --output=$SCRATCH/bashlogs/out/testbatch.%a.out
#SBATCH --error=$SCRATCH/bashlogs/err/testbatch.%a.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ncpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --mem=16G
#SBATCH --partition=giocomo,owners
#SBATCH -a 1-8

ml python/3.6.1

python3 pyClusterBatch.py -t $SLURM_ARRAY_TASK_ID -f PreProcessingTable_Li_11_9_2018.json
