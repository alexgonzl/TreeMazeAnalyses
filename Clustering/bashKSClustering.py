#!/usr/bin/env python

import os, json,sys, time, datetime
import numpy as np
from pathlib import Path

job_directory = Path("./.job")
job_directory.mkdir(parents=True, exist_ok=True)


ID = 'Al'
date = '4_17_2019'
overwriteFlag=1

date_obj = datetime.date.today()
date_str= "%s_%s_%s" % (date_obj.month,date_obj.day,date_obj.year)

table = "Clustering_{}_{}.json".format(ID,date)
TasksDir = Path.cwd()/'TasksDir'
if not TasksDir.exists():
    sys.exit('Task directory not found.')

if (TasksDir/table).exists():
    with open(str(TasksDir/table), 'r') as f:
        task_table = json.load(f)

nJobs = len(task_table)
for t in np.arange(1,nJobs+1):
    job_file = os.path.join(job_directory,"{}_t{}.job".format(ID,t))

    with open(job_file,"w+") as fh:
        fh.writelines("#!/bin/bash\n\n")
        fh.writelines("ml matlab/R2018a\n")
        fh.writelines("#SBATCH --job-name={}_t{}.job\n".format(ID,t))
        fh.writelines("#SBATCH --time=24:00:00\n")
        fh.writelines("$SBATCH --mail-user=alexg8@stanford.edu\n")
        fh.writelines("#SBATCH --mail-type=ALL\n\n")
        fun = "try matlabSherlockBashSession(%s,'%s'); catch; end; quit" % (t,table)
        fh.writelines('matlab -nojvm -r "%s" \n' % (fun))

    os.system("sbatch --partition=giocomo,owners --output=.job/t{}.o --error=.job/t{}.e --mem=8000 --cpus-per-task=2 {}".format(t,t,job_file))
    time.sleep(0.5)
