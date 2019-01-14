#!/usr/bin/env python

import os, json,sys, time, datetime
import numpy as np
from pathlib import Path

job_directory = Path("./.job")
job_directory.mkdir(parents=True, exist_ok=True)

ID = 'Li'
date = '1_10_2019'
overwriteFlag=0

date_obj = datetime.date.today()
date_str= "%s_%s_%s" % (date_obj.month,date_obj.day,date_obj.year)

table = "PreProcessingTable_{}_{}.json".format(ID,date)
TasksDir = Path.cwd()/'TasksDir'
if not TasksDir.exists():
    sys.exit('Task directory not found.')

if (TasksDir/table).exists():
    with open(str(TasksDir/table), 'r') as f:
        task_table = json.load(f)

nJobs = len(task_table)
completed_table = "PreProcessingTable_{}_{}_Completed.json".format(ID,date)
if not (TasksDir/completed_table).exists() or overwriteFlag:
    table_c = {}
    jobs = np.arange(1,nJobs+1)
    for t in jobs:
        table_c[str(t)] = 0
    table_c['table'] = table
    table_c['updated'] = date_str

    with open(str(TasksDir/completed_table), 'w') as f:
        json.dump(table_c, f ,indent=4)
else:
    with open(str(TasksDir/completed_table), 'r') as f:
        table_c =json.load(f)
    jobs = []
    for t in np.arange(1,nJobs+1):
        if not table_c[str(t)]:
            jobs.append(t)
    jobs = np.asarray(jobs)

for t in jobs:
#for t in [1,2]:
    job_file = os.path.join(job_directory,"{}_t{}.job".format(ID,t))

    with open(job_file,"w+") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("ml python/3.6.1\n")
        fh.writelines("#SBATCH --job-name={}_t{}.job\n".format(ID,t))
        fh.writelines("#SBATCH --error=.out/{}_t{}.err\n".format(ID,t))
        fh.writelines("#SBATCH --time=08:00:00\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("python3 pySherlockBatch_Session.py -t {} -f {}\n".format(t,table))

    os.system("sbatch --partition=giocomo,owners --mem=8000 --cpus-per-task=2 --mail-user=alexg8@stanford.edu {}".format(job_file))
    time.sleep(0.5)
