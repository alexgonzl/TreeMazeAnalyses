from pathlib import Path
import sys, getopt
import json, time

taskDir = Path('./TasksDir/')
if not taskDir.exists():
    sys.exit('Task directory not found.')

# Store taskID and TaskFile
taskID=-1
taskIDstr=''
taskFile=''

myopts, args = getopt.getopt(sys.argv[1:],"t:f:")
for o, a in myopts:
    if o == '-t':
        taskID=int(a)
        taskIDstr=str(a)
    elif o == '-f':
        taskFile = str(a)
    else:
        print("Usage: %s -t taskID -f tasks.json" % sys.argv[0])
        sys.exit('Invalid input. Aborting.')

try:
    with open(str(TaskDir / taskFile), 'r') as f:
            task_table = json.load(f)
except:
    sys.exit('Could not get Task Table. Aborting.')

try:
    t1=time.time()
    task = task_table[taskIDstr]
    task_type = task['type']
    if task_type=='tt':
        print("Processing Tetrode # {}".format(task['tt_id']))
        from pre_process_tetrode import get_process_save_tetrode
        get_process_save_tetrode(task)
    elif task_type == 'ev':
        print("Processing Events")
        from pre_process_tetrode import get_save_events
        get_save_events(task)
    elif task_type == 'vt':
        print("Processing Tracking Positions")
        from pre_process_tetrode import get_save_tracking
        get_save_tracking(task)
    t2=time.time()
    print("Task Completed. Total Task Time" % (t2-t1))
except:
    sys.exit('Error processing task {} of {}'.format(taskID,taskFile))
