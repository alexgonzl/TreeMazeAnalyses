import os, json,sys, time, datetime
import numpy as np
from pathlib import Path
from shutil import copy2

job_directory = Path("./.job")
job_directory.mkdir(parents=True, exist_ok=True)


ID = 'Li'
date = '3_13_2019'
overwriteFlag=1

date_obj = datetime.date.today()

table = "Clustering_{}_{}.json".format(ID,date)
TasksDir = Path.cwd()/'TasksDir'
if not TasksDir.exists():
    sys.exit('Task directory not found.')
    
if (TasksDir/table).exists():
    with open(str(TasksDir/table), 'r') as f:
        task_table = json.load(f)

nJobs = len(task_table)
for t in np.arange(1,nJobs+1):
    session = task_table[str(t)]['Files']
    nFiles = len(session)
    print()
    for f in np.arange(1,nFiles+1):
        copy2(session[str(f)]['filenames'],session[str(f)]['sp'])
        print('Copy complted for file {}'.format(session[str(f)]['filenames']))
        
    
