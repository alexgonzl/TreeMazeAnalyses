import json
from pathlib import Path
import datetime, sys, getopt
import numpy as np

def session_entry(session_name,Files,sp):
    return {'session_name':str(session_name), 'Files':Files, 'nFiles':len(Files),'sp':str(sp)}

def dict_entry(type,fn,sp,tt=-1):
    if type=='tt':
        return {'type':type,'filenames':fn,'tt_id':str(tt),'sp':str(sp)}
    else:
        return {'type':type,'filenames':str(fn),'sp':str(sp)}

if __name__ == '__main__':
    # Store taskID and TaskFile
    AnimalID=''
    volumePath=''
    minFileSize = 16384

    if len(sys.argv)<3:
        print("Usage: %s -a AnimalID -v 'Volume/path/to/folders'" % sys.argv[0])
        sys.exit('Invalid input.')

    myopts, args = getopt.getopt(sys.argv[1:],"a:v:")
    for o, a in myopts:
        print(o,a)
        if o == '-a':
            AnimalID=str(a)
        elif o == '-v':
            volumePath = Path(str(a))
            print(type(volumePath))
        else:
            print("Usage: %s -a AnimalID -v 'Volume/path/to/folders'" % sys.argv[0])
            sys.exit('Invalid input. Aborting.')

    TasksDir = Path('./TasksDir')
    TasksDir.mkdir(parents=True, exist_ok=True)

    date_obj = datetime.date.today()
    date_str= "%s_%s_%s" % (date_obj.month,date_obj.day,date_obj.year)

    Sessions = {}
    SessionCnt = 0
    for session in volumePath.glob('*_*[0-9]'):
        SessionCnt+=1
        print('Collecting Info for Session # {}, {}'.format(SessionCnt, session.name))
        sp = Path(str(session)+'_Results')
        sp.mkdir(parents=True, exist_ok=True)
        Files = {}
        taskID = 1
        try:
            # Look for valid records e.g. CSC1d.ncs
            for tt in np.arange(1,17):
                chAbsentFlag = False
                # check if there is any channel absent.
                TT = []
                for ch in ['a','b','c','d']:
                    try:
                        csc = 'CSC{}{}.ncs'.format(tt,ch)
                        if not (session / csc).exists():
                            chAbsentFlag = True
                        elif not (session / csc).stat().st_size>minFileSize:
                            chAbsentFlag = True
                        else:
                            chAbsentFlag = False
                            TT.append(str(session /csc))

                    except:
                        chAbsentFlag = True
                        print('Could not assign task to {}'.format(csc))
                        print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
                        continue

                if not chAbsentFlag:
                    Files[taskID] = dict_entry('tt',TT,sp,tt)
                    taskID+=1

            # If invalid record
            for csc in session.glob('CSC[0-9]*[a-d]_*.ncs'):
                try:
                    if csc.stat().st_size>minFileSize:
                        if csc.match('CSC'+str(tt)+ch +'.ncs'):
                            Files[taskID] = dict_entry('-tt_ch',csc)
                            taskID+=1
                except:
                    print('Could not assign task to {}'.format(csc))
                    continue

            # valid vt
            for vt in session.glob('*.nvt'):
                try:
                    if vt.stat().st_size>minFileSize:
                        #print(csc)
                        if vt.match('VT1.nvt'):
                            Files[taskID] = dict_entry('vt',vt,sp)
                            taskID+=1
                        elif vt.match('VT*.nvt'):
                            Files[taskID] = dict_entry('-vt',vt,sp)
                            taskID+=1
                except:
                    print('Could not assign task to {}'.format(vt))
                    continue

            # valid events
            for ev in session.glob('*.nev'):
                try:
                    if ev.stat().st_size>minFileSize:
                        if ev.match('Events.nev'):
                            Files[taskID] = dict_entry('ev',ev,sp)
                            taskID+=1
                        elif ev.match('Events_*.nev'):
                            Files[taskID] = dict_entry('-ev',ev,sp)
                            taskID+=1
                except:
                    print('Could not assign task to {}'.format(ev))
                    continue
            if len(Files)>0:
                Sessions[SessionCnt] = session_entry(session,Files,sp)
            else:
                print('Empty Session {}, discarding.'.format(str(session)))
                SessionCnt-=1
        except:
            print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
            continue

    print('Number of Sessions to be proccess = {}'.format(SessionCnt))
    with open(str(TasksDir)+'/PreProcessingTable_{}_{}.json'.format(AnimalID,date_str), 'w') as f:
        json.dump(Sessions, f ,indent=4)
