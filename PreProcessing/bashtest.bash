#!/bin/bash

ml python/3.6.1
for t in {2..4} 
do
    srun --partition=giocomo --output=$SCRATCH/bashlogs/out/testbatch%A.o \
	--cpus-per-task=4 --mem=8G python3 pyClusterBatch_Session.py -t $t -f PreProcessingTable_Li_11_13_2018.json
done

