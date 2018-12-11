#!/usr/bin/env python

import os, json,sys
from pathlib import Path

job_directory = Path("./.job")
job_directory.mkdir(parents=True, exist_ok=True)

ID = 'Al'
date = '12_10_2018'

table = "PreProcessingTable_{}_{}.json".format(ID,date)
TasksDir = Path.cwd()/'TasksDir'
if not TasksDir.exists():
    sys.exit('Task directory not found.')

if (TasksDir/table).exists():
    with open((str(TasksDir/table)), 'r') as f:
        task_table = json.load(f)

nJobs = len(task_table)

for t in range(1,nJobs+1):

    job_file = os.path.join(job_directory,"{}_t{}.job".format(ID,t))

    with open(job_file,"w+") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("ml python/3.6.1\n")
        fh.writelines("#SBATCH --job-name={}_t{}.job\n".format(ID,t))
        fh.writelines("#SBATCH --error=.out/{}_t{}.err\n".format(ID,t))
        fh.writelines("#SBATCH --time=04:00:00\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("python3 pySherlockBatch_Session.py -t {} -f {}\n".format(t,table))

    os.system("sbatch --partition=giocomo,owners --mem=8000 --cpus-per-task=2 --mail-user=alexg8@stanford.edu {}".format(job_file))
