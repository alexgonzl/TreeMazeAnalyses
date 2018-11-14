#!/bin/bash

#SBATCH --job-name=testbatch
#SBATCH --workdir=$HOME/TreeMazeAnalyses/PreProcessing
#SBATCH --output=$SCRATCH/bashlogs/out/testbatch_%A_%a.out
#SBATCH --error=$SCRATCH/bashlogs/err/testbatch_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --partition=giocomo
#SBATCH -a 1-44%8

ml python/3.6.1
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
python3 pyClusterBatch_Session.py -t $SLURM_ARRAY_TASK_ID -f PreProcessingTable_Li_11_13_2018.json
