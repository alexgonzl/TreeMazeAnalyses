import json
from pathlib import Path
import datetime, sys, getopt
import numpy as np
import shutil

def session_entry(session_name,Files,sp):
    return {'session_name':str(session_name), 'Files':Files, 'nFiles':len(Files),'sp':str(sp)}

def dict_entry(type,fn,hfn,sp):
    return {'type':type,'filenames':str(fn),'headerFile':str(hfn),'sp':str(sp)}

if __name__ == '__main__':
    # Store taskID and TaskFile
    volumePath=''
    ID = ''
    minFileSize = 16384

    if len(sys.argv)<4:
        print("Usage: %s -a ID -v 'Volume/path/to/folders' -p " % sys.argv[0])
        sys.exit('Invalid input.')

    print(sys.argv[1:])
    myopts, args = getopt.getopt(sys.argv[1:],"a:v:")
    for o, a in myopts:
        print(o,a)
        if o == '-v':
            volumePath = Path(str(a))
        elif o=='-a':
            ID = str(a)
        else:
            print("Usage: %s -a ID -v 'Volume/path/to/folders'" % sys.argv[0])
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
        taskID = 1 
        try:
            for tt in np.arange(1,17):
                file = 'tt_' + str(tt) + '.bin'
                sp = Path(str(session).strip('_Results')+'_KSClusters/tt_'+str(tt))
                sp.mkdir(parents=True,exist_ok=True)
                if (session / file).exists():
                    hfile = 'header_tt_' + str(tt)+'.json'
                    Files[taskID] = dict_entry('KiloSortCluster',str(session / file),str(session / hfile),sp)
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
    with open(str(TasksDir)+'/Clustering_{}_{}.json'.format(ID,date_str), 'w') as f:
        json.dump(Sessions, f ,indent=4)
