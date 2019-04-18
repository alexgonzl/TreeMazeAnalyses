import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
from pathlib import Path
import sys, getopt
import json, time, datetime

# Store taskID and TaskFile
taskID=-1
taskIDstr=''
taskFile=''
overwriteFlag=1

date_obj = datetime.date.today()
date_str= "%s_%s_%s" % (date_obj.month,date_obj.day,date_obj.year)

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
        taskFile = a
        taskFilePath = TasksDir/taskFile
    else:
        print("Usage: %s -t taskID -f tasks.json" % sys.argv[0])
        sys.exit('Invalid input. Aborting.')
try:
    if taskFilePath.exists():
        with open(str(taskFilePath), 'r') as f:
            task_table = json.load(f)
    else:
        sys.exit('Could not get Task Table. Aborting.')
except:
    sys.exit('Could not get Task Table. Aborting.')

try:
    tableCPath = taskFilePath.parent / (taskFilePath.stem + '_Completed.json')
    with open(str(tableCPath), 'r') as f:
        table_c = json.load(f)
except:
    print('Could load job completation table.')

session = task_table[taskIDstr]
nFiles = session['nFiles']
task_list = session['Files']
print("Processing Session {}".format(session['session_name']))
#if not table_c[taskIDstr] or overwriteFlag:
if 1:
    for file in task_list.keys():
        try:
            task=task_list[file]
            t1=time.time()
            task_type = task['type']
            if task_type=='tt':
                print("Processing Tetrode # {}, subSessionID={}".format(task['tt_id'],task['subSessionID']))
                from pre_process_neuralynx import get_process_save_tetrode
                get_process_save_tetrode(task,overwriteFlag=overwriteFlag)
            elif task_type == 'ev':
                print("Processing Events, subSessionID={}".format(task['subSessionID']))
                from pre_process_neuralynx import get_save_events
                get_save_events(task,overwriteFlag=overwriteFlag)
            elif task_type == 'vt':
                print("Processing Video Tracking, subSessionID={}".format(task['subSessionID']))
                from pre_process_neuralynx import get_save_tracking
                get_save_tracking(task,overwriteFlag=overwriteFlag)
            elif task_type == 'npy2bin':
                print("Processing Data Conversion to Binary")
                from dataConvert import npy2bin
                npy2bin(task['filenames'],task['sp'],overwriteFlag=overwriteFlag)

            t2=time.time()
            print("Task Completed. Total Task Time {}".format(t2-t1))
        except:
            print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
            print ("Unable to process task {} of {}".format(taskID,taskFile))
            #sys.exit('Error processing task {} of {}'.format(taskID,taskFile))            

try:
    table_c[taskIDstr] = 1
    table_c['updated'] = date_str
    with open(str(tableCPath), 'w') as f:
        json.dump(table_c, f ,indent=4)
except:
    print('Could not update job completation table.')
