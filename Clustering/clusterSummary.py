import sys,os, json, datetime, getopt, shutil
from pathlib import Path
import numpy as np
import pandas as pd

## Fixed Parameters
overwrite = 1
nTTs = 16

def addClusterEntry(pathToSummary, pathToSession, sessionName):

    pathToSummary = Path(pathToSummary)
    pathToSession = Path(pathToSession)

    assert pathToSession.exists(), 'Could not find session. Check Spelling/Path.'

    # Extract File Info
    tmp = sessionName.split('_')
    animal = tmp[0]
    task = tmp[1]
    date = tmp[2]

    Cl_Summary_fn = pathToSummary
    if Cl_Summary_fn.exists():
        with Cl_Summary_fn.open(mode='r') as f:
            cluster_summary = json.load(f)
        if not date in cluster_summary[animal].keys() or overwrite:
            cluster_summary[animal][date]={}
        if not task in cluster_summary[animal][date].keys() or overwrite:
            cluster_summary[animal][date][task] = {}
        else:
            print('Warning clustering data exists for {}, and overwrite = false'.format(sessionName))
    else:
        cluster_summary = {}
        cluster_summary[animal] = {}
        cluster_summary[animal][date]={}
        cluster_summary[animal][date][task] = {}
        print('Warning. Summary Json File not found, creating one.')

        cluster_summary[animal][date][task]['nCells'] = 0
        cluster_summary[animal][date][task]['nMua'] = 0
        cluster_summary[animal][date][task]['cell_IDs'] = {}
        cluster_summary[animal][date][task]['mua_IDs'] = {}
        for tt in np.arange(1,nTTs+1):
            fn = pathToSession/('tt_'+str(tt))/'cluster_group.tsv'
            assert fn.exists(), 'could not find record for tt {}; in {}'.format(tt,sessionName)

            cluster_summary[animal][date][task]['cell_IDs'][int(tt)]=[]
            cluster_summary[animal][date][task]['mua_IDs'][int(tt)]=[]

            d=pd.read_csv(fn,delimiter='\t')
            cells = np.where(d['group']=='good')[0].tolist()
            mua = np.where(d['group']=='mua')[0].tolist()

            cluster_summary[animal][date][task]['cell_IDs'][int(tt)]=[]
            cluster_summary[animal][date][task]['mua_IDs'][int(tt)]=[]
            for cc in cells:
                cluster_summary[animal][date][task]['cell_IDs'][int(tt)].append(cc)
            for mm  in mua:
                cluster_summary[animal][date][task]['mua_IDs'][int(tt)].append(mm)
            cluster_summary[animal][date][task]['nCells'] += len(cells)
            cluster_summary[animal][date][task]['nMua'] += len(mua)

        with pathToSummary.open(mode='w') as f:
                json.dump(cluster_summary, f, indent=4)

        n=0
        m=0
        for d in cluster_summary[animal].keys():
            n+=cluster_summary[animal][d][task]['nCells']
            m+=cluster_summary[animal][d][task]['nMua']
        print(" nSessions = {} \n nCells = {} \n nMua = {}".format(len(cluster_summary[animal].keys()),n,m))

def CopyClustersToOak(localDir,oakDir):
    localDir = Path(localDir)
    oakDir = Path(oakDir)
    for localSummary in localDir.glob("*_ClusteringSummary.json"):
        SummaryName = localSummary.name
        fullPathFN = str(localSummary)

        if not filecmp.cmp(fullPathFN,str(oakDir/SummaryName),shallow=True):
            shutil.copyfile(fullPathFN,str(oakDir/SummaryName))
            print('updated file summary file')
        else:
            print('summary file is the same, skipping copy.')

        with fullPathFN.open(mode='r') as f:
            cluster_summary = json.load(f)

        for animal in cluster_summary.keys():
            # copy individual tetrode clusters
            notUpDatedList = []
            for date in cluster_summary[animal].keys():
                for task in cluster_summary[animal][date].keys():
                    sessionName = animal+'_'+task+'_'+date+'_KSClusters'
                    notUpDatedList = []
                    for tt in np.arange(1,nTTs+1):
                        fn = localDir/sessionName/('tt_'+str(tt))/'cluster_group.tsv'
                        sp = oakDir/sessionName/('tt_'+str(tt))/'cluster_group.tsv'
                        if not filecmp.cmp(str(fn),str(sp),shallow=True):
                            shutil.copyfile(str(fn),str(sp))
                        else:
                            notUpDatedList.append(tt)
                    if len(notUpDatedList)==16:
                        print("{}: All tetrodes have already been clustered. ".format(sessionName))
                    elif len(notUpDatedList)==0:
                        print("{}: Updated all tetrode clusters".format(sessionName))
                    else:
                        print("{}: Indetical cluster files, no updates for TTs {}".format(sessionName, notUpDatedList))

def UpdateClusterTable()
