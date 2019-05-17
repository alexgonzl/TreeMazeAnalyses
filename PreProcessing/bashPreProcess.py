#!/usr/bin/env python

import os, json,sys, time, datetime
import numpy as np
from pathlib import Path

job_directory = Path("./.job")
job_directory.mkdir(parents=True, exist_ok=True)

ID = 'P'
date = '5_17_2019'
overwriteFlag=1

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
#for t in [1]:
    job_file = os.path.join(job_directory,"{}_t{}.job".format(ID,t))

    with open(job_file,"w+") as fh:
        fh.writelines("#!/bin/bash\n\n")
        fh.writelines("#SBATCH --job-name={0}_t{1}.job\n".format(ID,t))
        fh.writelines("#SBATCH -e .job/{0}_t{1}.e \n".format(ID,t))
        fh.writelines("#SBATCH -o .job/{0}_t{1}.o \n".format(ID,t))
        fh.writelines("#SBATCH --mail-user=alexg8@stanford.edu\n")
        fh.writelines("#SBATCH --mail-type=BEGIN,END,FAIL\n")
        fh.writelines("#SBATCH --time=08:00:00 \n\n")
        
        fh.writelines("ml python/3.6\n")
        fh.writelines("python3 pySherlockBatch_Session.py -t {} -f {}\n".format(t,table))

    os.system("sbatch --partition=giocomo,owners --mem=32000 --cpus-per-task=4 {}".format(job_file))
    time.sleep(0.5)
