#!/bin/bash

ml python/3.6.1

for t in {1..44}; do

#SBATCH --job-name=testbatch${t}
#SBATCH --workdir=$HOME/TreeMazeAnalyses/PreProcessing
#SBATCH --output=$SCRATCH/bashlogs/out/testbatch_%_%t.out
#SBATCH --error=$SCRATCH/bashlogs/err/testbatch_%A_%t.err
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --partition=giocomo

sbatch python3 pyClusterBatch_Session.py -t ${t} -f PreProcessingTable_Li_11_13_2018.json

done
