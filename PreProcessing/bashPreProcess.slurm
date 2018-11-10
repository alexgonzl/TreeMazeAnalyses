#!/bin/bash

#SBATCH --job-name=testbatch 
#SBATCH --output=testbatch.%j.out
#SBATCH --error=testbatch.%j.err
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ncpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=giocomo,owners

ml python/3.6.1

srun python3 P
