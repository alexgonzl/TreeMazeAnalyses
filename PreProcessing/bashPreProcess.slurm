#!/bin/bash

#SBATCH --job-name=testbatch
#SBATCH --workdir=$HOME/TreeMazeAnalyses/PreProcessing
#SBATCH --output=$SCRATCH/bashlogs/out/testbatch_%A_%a.out
#SBATCH --error=$SCRATCH/bashlogs/err/testbatch_%A_%a.err
#SBATCH --ntasks=16
#SBATCH --nodes=2
#SBATCH --ncpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --partition=giocomo,owners
#SBATCH -a 1-16

ml python/3.6.1
python3 pyClusterBatch.py -t ${SLURM_ARRAY_TASK_ID} -f PreProcessingTable_Li_11_13_2018.json
