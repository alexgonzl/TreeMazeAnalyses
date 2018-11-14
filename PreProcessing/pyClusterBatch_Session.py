import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from pathlib import Path
import sys, getopt
import json, time

# Store taskID and TaskFile
taskID=-1
taskIDstr=''
taskFile=''

TasksDir = Path.cwd()/'TasksDir'
if not TasksDir.exists():
    sys.exit('Task directory not found.')

if len(sys.argv)<3:
    print("Usage: %s -t task# -f 'path/to/tasks.json'" % sys.argv[0])
    sys.exit('Invalid input.')

myopts, args = getopt.getopt(sys.argv[1:],"t:f:")
for o, a in myopts:
    if o == '-t':
        taskID=int(a)
        taskIDstr=str(a)
    elif o == '-f':
        taskFile = Path(a)
    else:
        print("Usage: %s -t taskID -f tasks.json" % sys.argv[0])
        sys.exit('Invalid input. Aborting.')

try:
    if (TasksDir/taskFile).exists():
        with open((str(TasksDir/taskFile)), 'r') as f:
                task_table = json.load(f)
    else:
        sys.exit('Could not get Task Table. Aborting.')
except:
    sys.exit('Could not get Task Table. Aborting.')

task = task_table[taskIDstr]
nFiles = task['nFiles']
for file in range(1,nFiles+1):
    try:
        t1=tt1=time.ti
        print(task)
        task_type = task['type']
        if task_type=='tt':
            print("Processing Tetrode # {}".format(task['tt_id']))
            from pre_process_neuralynx import get_process_save_tetrode
            get_process_save_tetrode(task)
        elif task_type == 'ev':
            print("Processing Events")
            from pre_process_neuralynx import get_save_events
            get_save_events(task)
        elif task_type == 'vt':
            print("Processing Tracking Positions")
            from pre_process_neuralynx import get_save_tracking
            get_save_tracking(task)
        t2=time.time()
        print("Task Completed. Total Task Time {}".format(t2-t1))
    except:
        print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
        sys.exit('Error processing task {} of {}'.format(taskID,taskFile))
