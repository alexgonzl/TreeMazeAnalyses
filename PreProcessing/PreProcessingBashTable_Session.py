import json
from pathlib import Path
import datetime, sys, getopt
import numpy as np

def session_entry(session_name,Files,sp,nSubSessions,validSubSessions):
    return {'session_name':str(session_name), 'Files':Files, 'nFiles':len(Files),
    'sp':str(sp),'nSubSessions':nSubSessions,'validSubSessions':validSubSessions.tolist()}

def dict_entry(type,fn,sp,subSessionID,tt=-1):
    if type=='tt':
        return {'type':type,'filenames':fn,'tt_id':str(tt),'sp':str(sp),'subSessionID':str(subSessionID)}
    else:
        return {'type':type,'filenames':fn,'sp':str(sp),'subSessionID':str(subSessionID)}

if __name__ == '__main__':
    # Store taskID and TaskFile
    AnimalID=''
    volumePath=''
    minFileSize = 16384
    TetrodeRecording = 1
    nTetrodes = 16

    if len(sys.argv)<3:
        print("Usage: %s -a AnimalID -v 'Volume/path/to/folders'" % sys.argv[0])
        sys.exit('Invalid input.')

    myopts, args = getopt.getopt(sys.argv[1:],"a:v:p:")
    for o, a in myopts:
        if o == '-a':
            AnimalID=str(a)
        elif o == '-v':
            volumePath = Path(str(a))
        elif o == '-p':
            if str(a)=='NR32':
                TetrodeRecording = 0
                nChannels = 32
            elif str(a)=='TT16':
                TetrodeRecording = 1
                nTetrodes=16
            else:
                sys.exit('Invalid Probe Type.')
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
        nSubSessions = len(list(session.glob('VT1*.nvt')))
        validSubSessions = np.ones(nSubSessions,dtype=bool)
        taskID = 1
        try:
            # Look for valid records e.g. CSC1d.ncs
            if TetrodeRecording:
                for tt in np.arange(1,nTetrodes+1):
                    for ss in np.arange(nSubSessions):
                        if TetrodeRecording:
                            TT=[]
                            chAbsentFlag = False
                            for ch in ['a','b','c','d']:
                                try:
                                    if ss ==0:
                                        csc = 'CSC{}{}.ncs'.format(tt,ch)
                                    else:
                                        csc = 'CSC{}{}_{}.ncs'.format(tt,ch,str(ss).zfill(4))

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
                                Files[taskID] = dict_entry('tt',TT,sp,tt=tt,subSessionID=str(ss).zfill(4))
                                taskID+=1
            else:
                
                for ss in np.arange(nSubSessions):
                    Probe =[]
                    chAbsentFlag=False
                    for ch in np.arange(1,nChannels+1):
                        try:
                            if ss==0:
                                csc = 'CSC{}.ncs'.format(ch)
                            else:
                                csc = 'CSC{}_{}.ncs'.format(ch,str(ss).zfill(4))
                                
                            if not (session / csc).exists(): # file does not exists
                                chAbsentFlag = True
                            elif not (session / csc).stat().st_size>minFileSize: # file exists but it is empty
                                chAbsentFlag = True
                            else: # file exists and its valid
                                chAbsentFlag = False
                                Probe.append(str(session /csc))
                        except:
                            chAbsentFlag = True
                            #print('Could not assign task to {}'.format(csc))
                            print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
                            continue
                    if not chAbsentFlag:
                        Files[taskID] = dict_entry('probe',Probe,sp,subSessionID=str(ss).zfill(4))
                        taskID+=1

            # valid vt
            for vt in session.glob('*.nvt'):
                try:
                    if vt.stat().st_size>minFileSize:
                        if vt.match('VT1.nvt'):
                            Files[taskID] = dict_entry('vt',str(vt),sp,subSessionID='0000')
                            taskID+=1
                        else:
                            for ss in np.arange(1,nSubSessions):
                                ss_str = str(ss).zfill(4)
                                if vt.match('VT1_{}.nvt'.format(ss_str)):
                                    Files[taskID] = dict_entry('vt',str(vt),sp,subSessionID=ss_str)
                                    taskID+=1
                    else:
                        validSubSessions[ss]=False
                except:
                    print('Could not assign task to {}'.format(vt))
                    print(sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
                    continue

            # valid events
            for ev in session.glob('*.nev'):
                try:
                    if ev.stat().st_size>minFileSize:
                        if ev.match('Events.nev'):
                            Files[taskID] = dict_entry('ev',str(ev),sp,subSessionID='0000')
                        else:
                            for ss in np.arange(1,nSubSessions):
                                ss_str = str(ss).zfill(4)
                                if ev.match('Events_{}.nev'.format(ss_str)):
                                    Files[taskID] = dict_entry('ev',str(ev),sp,subSessionID=ss_str)
                                    taskID+=1
                except:
                    print('Could not assign task to {}'.format(ev))
                    print(sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
                    continue
            if len(Files)>0:
                Sessions[SessionCnt] = session_entry(session,Files,sp,nSubSessions,validSubSessions)
            else:
                print('Empty Session {}, discarding.'.format(str(session)))
                SessionCnt-=1
        except:
            print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
            continue

    print('Number of Sessions to be proccess = {}'.format(SessionCnt))
    with open(str(TasksDir)+'/PreProcessingTable_{}_{}.json'.format(AnimalID,date_str), 'w') as f:
        json.dump(Sessions, f ,indent=4)
