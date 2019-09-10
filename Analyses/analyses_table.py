# creates a table with progress of analyses for each session
# update analyses table

from pathlib import Path
import os,sys, json, datetime, getopt
import pickle as pkl
import time
import pandas as pd
import numpy as np

sys.path.append('../PreProcessing/')
sys.path.append('../Lib/')
sys.path.append('../Analyses/')

import TrialAnalyses as TA
import TreeMazeFunctions as TMF

nTetrodes=16

def getOakPaths():
    oakPaths = {}
    oakPaths['Root'] = Path('/mnt/o/giocomo/alexg/')
    oakPaths['Clustered'] = Path('/mnt/o/giocomo/alexg/Clustered/')
    oakPaths['PreProcessed'] = Path('/mnt/o/giocomo/alexg/PreProcessed/')
    oakPaths['Raw'] = Path('/mnt/o/giocomo/alexg/RawData/InVivo/')
    oakPaths['Analyses'] = Path('/mnt/o/giocomo/alexg/Analyses')
    return oakPaths

def getAnimalPaths(rootPath,animal):
    Paths = {}
    Paths['Clusters'] = rootPath['Clustered'] / animal
    Paths['Raw'] = rootPath['Raw'] / animal
    Paths['PreProcessed'] = rootPath['PreProcessed']
    Paths['Analyses'] = rootPath['Analyses'] / animal
    return Paths

def getSessionPaths(rootPath, session,step=0.02,SR=32000):
    tmp = session.split('_')
    animal = tmp[0]
    task = tmp[1]
    date = tmp[2]

    Paths = {}
    Paths['session'] = session
    Paths['animal']=animal
    Paths['task'] = task
    Paths['date'] = date
    Paths['step'] = step
    Paths['SR'] = SR
    Paths['Clusters'] = rootPath['Clustered'] / animal /(session+'_KSClusters')
    Paths['Raw'] = rootPath['Raw'] / animal / session
    Paths['PreProcessed'] = rootPath['PreProcessed'] / animal / (session + '_Results')
    Paths['Analyses'] = rootPath['Analyses'] / animal/ (session + '_Analyses')

    Paths['ClusterTable'] = rootPath['Clustered'] / animal / (animal+'_ClusteringSummary.json')

    Paths['Analyses'].mkdir(parents=True, exist_ok=True)

    Paths['BehavTrackDat'] = Paths['Analyses'] / ('BehTrackVariables_{}ms.h5'.format(int(step*1000)))

    for ut in ['Cell','Mua']:
        Paths[ut + '_Spikes'] = Paths['Analyses'] / (ut+'_Spikes.json')
        Paths[ut + '_WaveForms'] = Paths['Analyses'] / (ut+'_WaveForms.pkl')
        Paths[ut + '_WaveFormInfo'] = Paths['Analyses'] / (ut+'_WaveFormInfo.pkl')
        Paths[ut + '_Bin_Spikes'] = Paths['Analyses'] / ('{}_Bin_Spikes_{}ms.npy'.format(ut,int(step*1000)))
        Paths[ut + '_FR'] = Paths['Analyses'] / ('{}_FR_{}ms.npy'.format(ut,int(step*1000)))

    Paths['Spike_IDs'] = Paths['Analyses'] / 'Spike_IDs.json'
    Paths['ZoneAnalyses'] = Paths['Analyses'] / 'ZoneAnalyses.pkl'

    Paths['TrialInfo'] = Paths['Analyses'] / 'TrInfo.pkl'
    Paths['TrialCondMat'] = Paths['Analyses'] / 'TrialCondMat.csv'
    Paths['TrLongPosMat'] = Paths['Analyses'] / 'TrLongPosMat.csv'
    Paths['TrLongPosFRDat'] = Paths['Analyses'] / 'TrLongPosFRDat.csv'
    Paths['TrModelFits'] = Paths['Analyses'] /  'TrModelFits.csv'

    Paths['CueDesc_SegUniRes'] = Paths['Analyses'] / 'CueDesc_SegUniRes.csv'
    Paths['CueDesc_SegDecRes'] = Paths['Analyses'] / 'CueDesc_SegDecRes.csv'
    Paths['CueDesc_SegDecSumRes'] = Paths['Analyses'] / 'CueDesc_SegDecSumRes.csv'
    Paths['PopCueDesc_SegDecSumRes'] = Paths['Analyses'] / 'PopCueDesc_SegDecSumRes.csv'

    # plots directories
    Paths['Plots'] = Paths['Analyses'] / 'Plots'
    Paths['Plots'].mkdir(parents=True, exist_ok=True)
    Paths['SampCountsPlots'] = Paths['Plots'] / 'SampCountsPlots'
    Paths['SampCountsPlots'].mkdir(parents=True, exist_ok=True)

    Paths['ZoneFRPlots'] = Paths['Plots'] / 'ZoneFRPlots'
    Paths['ZoneFRPlots'].mkdir(parents=True, exist_ok=True)

    Paths['ZoneCorrPlots'] = Paths['Plots'] / 'ZoneCorrPlots'
    Paths['ZoneCorrPlots'].mkdir(parents=True, exist_ok=True)
    Paths['SIPlots'] = Paths['Plots'] / 'SIPlots'
    Paths['SIPlots'].mkdir(parents=True, exist_ok=True)

    Paths['TrialPlots'] = Paths['Plots'] / 'TrialPlots'
    Paths['TrialPlots'].mkdir(parents=True, exist_ok=True)

    Paths['CueDescPlots'] = Paths['Plots'] / 'CueDescPlots'
    Paths['CueDescPlots'].mkdir(parents=True, exist_ok=True)

    return Paths

def checkRaw(sePaths,aTable):

    for se in aTable.index:
        rawFlag = 1
        for ch in ['a','b','c','d']:
            for tt in np.arange(1,nTetrodes+1):
                if not (sePaths[se]['Raw'] / ('CSC{}{}.ncs'.format(tt,ch))).exists():
                    rawFlag = 0
                    break
        aTable.loc[se,'Raw']=rawFlag
    return aTable

def checkPrePro(sePaths,aTable):
    for se in aTable.index:
        allTTFlag = 1
        partialFlag = 0
        for tt in np.arange(1,nTetrodes+1):
            if not (sePaths[se]['PreProcessed'] / ('tt_{}.bin'.format(tt))).exists():
                allTTFlag=0
            else:
                partialFlag=1

        aTable.loc[se,'PP']=partialFlag
        aTable.loc[se,'PP_A']=allTTFlag
    return aTable

def checkSort(sePaths,aTable):
    for se in aTable.index:
        allTTFlag = 1
        partialFlag = 0
        for tt in np.arange(1,nTetrodes+1):
            if not (sePaths[se]['Clusters'] / ('tt_{}'.format(tt)) / 'rez.mat').exists():
                allTTFlag=0
            else:
                partialFlag=1

        aTable.loc[se,'Sort']=partialFlag
        aTable.loc[se,'Sort_A']=allTTFlag
    return aTable

def checkClust(sePaths,aTable):
    for se in aTable.index:
        allTTFlag = 1
        partialFlag = 0
        for tt in np.arange(1,nTetrodes+1):
            if not (sePaths[se]['Clusters'] / ('tt_{}'.format(tt)) / 'cluster_group.tsv').exists():
                allTTFlag=0
            else:
                partialFlag=1

        aTable.loc[se,'Clust']=partialFlag
        aTable.loc[se,'Clust_A']=allTTFlag
    return aTable

def checkFR(sePaths,aTable):
    for se in aTable.index:
        allFR = 1
        partialFR = 0
        if not (sePaths[se]['Cell_Bin_Spikes']).exists():
            allFR = 0
        else:
            dat = np.load(sePaths[se]['Cell_Bin_Spikes'])
            if np.all(dat.sum(axis=1)>0):
                partialFR = 1
            else:
                allFR = 0

        aTable.loc[se,'FR']=partialFR
        aTable.loc[se,'FR_A']=allFR
    return aTable

def checkZoneAnalyses(sePaths,aTable):
    for se in aTable.index:
        if not (aTable.loc[se,'Task']=='OF'):
            if sePaths[se]['ZoneAnalyses'].exists():
                aTable.loc[se,'Zone']=1
            else:
                aTable.loc[se,'Zone']=0
    return aTable

def checkTrialAnalyses(sePaths,aTable):
    for se in aTable.index:
        if not (aTable.loc[se,'Task']=='OF'):
            if sePaths[se]['TrialCondMat'].exists():
                aTable.loc[se,'Trial']= 1
            else:
                aTable.loc[se,'Trial']= 0

            if sePaths[se]['TrModelFits'].exists():
                aTable.loc[se,'TrModels'] = 1
            else:
                aTable.loc[se,'TrModels'] = 0

    return aTable

def loadSessionData(sessionPaths):
    wfi = {}
    bin_spikes = {}
    fr = {}

    mods = {}
    params = TA.getParamSet()
    for k,pp in params.items():
        s =''
        for p in pp:
            s+='-'+p
        mods[k]=s[1:]

    for ut in ['Cell','Mua']:
        with sessionPaths[ut+'_WaveFormInfo'].open(mode='rb') as f:
            wfi[ut] = pkl.load(f)
        bin_spikes[ut]=np.load(sessionPaths[ut+'_Bin_Spikes'])
        fr[ut] = np.load(sessionPaths[ut+'_FR'])

    with sessionPaths['Spike_IDs'].open() as f:
        ids = json.load(f)
    with sessionPaths['ZoneAnalyses'].open(mode='rb') as f:
        za = pkl.load(f)

    PosDat = TMF.getBehTrackData(sessionPaths)

    TrialLongMat = pd.read_csv( sessionPaths['TrLongPosMat'],index_col=0)
    TrialFRLongMat = pd.read_csv(sessionPaths['TrLongPosFRDat'],index_col=0)
    fitTable = pd.read_csv(sessionPaths['TrModelFits'],index_col=0)
    TrialConds = pd.read_csv(sessionPaths['TrialCondMat'] ,index_col=0)


    if isinstance(fitTable,pd.core.frame.DataFrame):
        nUnits = fitTable.shape[0]
        x=[]
        for i in np.arange(nUnits):
            if np.isnan(fitTable['modelNum'][i]):
                x.append('UnCla')
            else:
                x.append(mods[fitTable['modelNum'][i]])
        fitTable['selMod'] = x

    dat = {}
    dat['wfi'] = wfi
    dat['bin_spikes'] = bin_spikes
    dat['fr'] = fr

    dat['ids'] = ids
    dat['za'] = za

    dat['PosDat'] = PosDat
    dat['TrialLongMat'] = TrialLongMat
    dat['TrialFRLongMat'] = TrialFRLongMat
    dat['TrialModelFits'] = fitTable
    dat['TrialConds'] = TrialConds

    return dat


# if __name__ == '__main__':
#     ID = ''
#     minFileSize = 16384
#     TetrodeRecording = 1
#     nTetrodes = 16
#
#     if len(sys.argv)<2:
#         print("Usage: %s -a ID " % sys.argv[0])
#         sys.exit('Invalid input.')
#
#     print(sys.argv[1:])
#     myopts, args = getopt.getopt(sys.argv[1:],"a:p:")
#     for o, a in myopts:
#         print(o,a)
#         if o == '-a':
#             ID = str(a)
#         elif o == '-p':
#             if str(a)=='NR32':
#                 TetrodeRecording = 0
#                 nChannels = 32
#             elif str(a)=='TT16':
#                 TetrodeRecording = 1
#                 nTetrodes=16
#             else:
#                 sys.exit('Invalid Probe Type.')
#         else:
#             print("Usage: %s -a ID " % sys.argv[0])
#             sys.exit('Invalid input. Aborting.')
#
#     oakPaths = getOakPaths()
