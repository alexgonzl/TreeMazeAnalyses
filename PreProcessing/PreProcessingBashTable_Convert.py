import json
from pathlib import Path
import datetime, sys, getopt
import numpy as np

# python script to find .npy files and convert them to binary (int16),
# and deleting the .npy files (saving %50 of space)
# refer to python notebook for some QA checks on this conversion.
def session_entry(session_name,Files,sp):
    return {'session_name':str(session_name), 'Files':Files, 'nFiles':len(Files),'sp':str(sp)}

def dict_entry(type,fn,sp):
    return {'type':type,'filenames':str(fn),'sp':str(sp)}

if __name__ == '__main__':
    # Store taskID and TaskFile
    volumePath=''
    minFileSize = 16384

    if len(sys.argv)<2:
        print("Usage: %s -v 'Volume/path/to/folders' -p " % sys.argv[0])
        sys.exit('Invalid input.')

    myopts, args = getopt.getopt(sys.argv[1:],"a:v:")
    for o, a in myopts:
        print(o,a)
        if o == '-v':
            volumePath = Path(str(a))
        else:
            print("Usage: %s -v 'Volume/path/to/folders'" % sys.argv[0])
            sys.exit('Invalid input. Aborting.')

    TasksDir = Path('./TasksDir')
    TasksDir.mkdir(parents=True, exist_ok=True)

    date_obj = datetime.date.today()
    date_str= "%s_%s_%s" % (date_obj.month,date_obj.day,date_obj.year)

    Sessions = {}
    SessionCnt = 0
    for session in volumePath.glob('*_Results'):
        SessionCnt+=1
        print('Collecting Info for Session # {}, {}'.format(SessionCnt, session.name))
        Files = {}
        taskID = 1 #
        try:
            for tt in np.arange(1,17):
                file = 'tt_' + str(tt) + '.npy'
                if (session / file).exists():
                    Files[taskID] = dict_entry('npy2bin',str(session / file),session)
                    taskID+=1
            if len(Files)>0:
                Sessions[SessionCnt] = session_entry(session,Files,session)
            else:
                print('Empty Session {}, discarding.'.format(str(session)))
                SessionCnt-=1
        except:
            print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
            continue

    print('Number of Sessions to be proccess = {}'.format(SessionCnt))
    with open(str(TasksDir)+'/PreProcessingTable_{}.json'.format(date_str), 'w') as f:
        json.dump(Sessions, f ,indent=4)
