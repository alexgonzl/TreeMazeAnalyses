import json
from pathlib import Path
import datetime
AnimalID = 'Li'
volumePath=Path('/scratch/users/alexg8/Data/'+AnimalID)

taskID = 1
nSessions = 0
Files = {}
TTs={}
date_obj = datatime.data.today()
date_str= "%s_%s_%s" % (date_obj.month,date_obj.day,date_obj.year)

for tt in np.arange(1,17):
    TTs[tt] = ['CSC{}{}.ncs'.format(tt,i) for i in ['a','b','c','d']]

for fp in volumePath.glob('*'+task+'*18'):
    try:
        nSessions+=1
        print('Collecting Info for Session '+str(fp.name))
        sp = Path(str(fp)+'_Results')
        sp.mkdir(parents=True, exist_ok=True)

        Files{taskID} = {'type':'ev','filename':'Events.nev','filepath':fp,'savepath':sp}
        taskID+=1
        Files{taskID} = {'type':'vt','filename':'VT1.nvt','filepath':fp,'savepath':sp}
        taskID+=1

        for tt in np.arange(1,17):
            Files{taskID} = {'type':'tt','filename':TTs[tt],'filepath':fp,'savepath':sp}
            taskID+=1

    except:
        print('Could not process file ' + fp.name)
        print ("error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
        continue

print('Number of Files to be processed = {}, for {} sessions'.format(taskID-1,nSessions))

with open('PreProcessingTable_{}_{}.json'.format(AnimalID,date_str), 'w') as f:
    json.dump(Files, f ,indent=4)
