import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats
from scipy.interpolate import CubicSpline
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import AnchoredText
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

import seaborn as sns
font = {'family' : 'sans-serif',
        'size'   : 20}
plt.rc('font', **font)
plt.rc('text',usetex=False)

from pathlib import Path
import os,sys, time
import h5py, json
import pickle as pkl

sys.path.append('../PreProcessing/')
sys.path.append('../TrackingAnalyses/')
sys.path.append('../Lib/')
sys.path.append('../Analyses/')
from filters_ag import *

from importlib import reload  # Python 3.4+ only.
import pre_process_neuralynx as PPN
import TreeMazeFunctions as TMF
import spike_functions as SF
import spatial_tuning as ST
import stats_functions as StatsF
import plot_functions as PF
import zone_analyses_session as ZA

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def getPosSequence(PosZones,startID,endID):
    nSamps = len(PosZones)
    pos = []
    samp = []

    pos.append(PosZones[0])
    samp.append(0)
    for p in np.arange(nSamps-1):
        p0 = PosZones[p]
        p1 = PosZones[p+1]
        if p0!=p1:
            pos.append(p1)
            samp.append(p+1)
    pos = np.array(pos)
    samp = np.array(samp) + startID
    nPos = len(pos)
    dur = np.zeros(nPos,dtype=int)
    for p in np.arange(nPos-1):
        dur[p] = samp[p+1]-samp[p]
    dur[-1] = endID-samp[-1]

    return pos, samp, dur

def cmp(a,b):
    return (a>b)-(a<b)

def getTrials(dat,**kwargs):
    nTr = dat.shape[0]
    trials = set(np.arange(1,nTr+1))
    try:
        for k,v in kwargs.items():
            trials = trials & set(np.where(dat[k]==v)[0]+1)
    except:
        print('Invalid Selection {} {}'.format(k,v))
        pass
    return np.sort(np.array(list(trials)))

def zscore(x,mu,sig):
    return (x-mu)/sig

def getFR_TrZone(TrInfo, FRMat):
    nCells = FRMat.shape[0]
    TrZnFR = {} # FR for every zone visited in that trial
    OTrZnFR = {} # FR for every zone visited in that trial

    for tr in TrInfo['All']['Trials']:
        nPos = len(TrInfo['TrSeq']['Pos'][tr])
        trSpPos = np.zeros((nCells,nPos))
        for p in np.arange(nPos):
            s=TrInfo['TrSeq']['Samp'][tr][p]
            d=TrInfo['TrSeq']['Dur'][tr][p]
            samps = np.arange(s,s+d)
            for cell in np.arange(nCells):
                trSpPos[cell, p]=np.mean(FRMat[cell][samps])

        nPos = len(TrInfo['OffTrSeq']['Pos'][tr])
        otrSpPos = np.zeros((nCells,nPos))
        for p in np.arange(nPos):
            s=TrInfo['OffTrSeq']['Samp'][tr][p]
            d=TrInfo['OffTrSeq']['Dur'][tr][p]
            samps = np.arange(s,s+d)
            for cell in np.arange(nCells):
                otrSpPos[cell, p]=np.mean(FRMat[cell][samps])

        TrZnFR[tr] = trSpPos
        OTrZnFR[tr] = otrSpPos
    return TrZnFR, OTrZnFR

oakPaths = {}
oakPaths['Root'] = Path('/mnt/o/giocomo/alexg/')
oakPaths['Clustered'] = oakPaths['Root']/'Clustered'
oakPaths['PreProcessed'] = oakPaths['Root']/'PreProcessed'
oakPaths['Raw'] =oakPaths['Root'] / 'RawData'/'InVivo'
oakPaths['Analyses'] =oakPaths['Root']/'Analyses'

################################################################################
ValidTraj = {'R_S1':['Home','SegA','Center','SegB','I1','SegC','G1'],
               'R_S2':['Home','SegA','Center','SegB','I1','SegD','G2'],
               'R_L1':['Home','SegA','Center','SegB','I1','SegD','G2','SegD','I1','SegC','G1'],
               'R_L2':['Home','SegA','Center','SegB','I1','SegC','G1','SegC','I1','SegD','G2'],
               'L_S3':['Home','SegA','Center','SegE','I2','SegF','G3'],
               'L_S4':['Home','SegA','Center','SegE','I2','SegG','G4'],
               'L_L3':['Home','SegA','Center','SegE','I2','SegG','G4','SegG','I2','SegF','G3'],
               'L_L4':['Home','SegA','Center','SegE','I2','SegF','G3','SegF','I2','SegG','G4'],
              }
ValidOffTraj = {}
for k,v in ValidTraj.items():
    ValidOffTraj[k] = v[::-1]

def main(session):
    # Load Data
    sessionPaths = ZA.getSessionPaths(oakPaths,session)
    PosDat = TMF.getBehTrackData(sessionPaths,0)
    cell_bin_spikes, mua_bin_spikes, ids= SF.getSessionBinSpikes(sessionPaths,PosDat['t'])
    cell_FR, mua_FR = SF.getSessionFR(sessionPaths)
    mu=np.mean(cell_FR,1)
    sig=np.std(cell_FR,1)

    # dictionary with all trial information
    TrInfo = getTrInfo(PosDat)
    # matrix with several trial conditions of interest
    TrCondMat = getTrCondMat(TrInfo)
    # long form matrix expanded by the animal positions in a trial
    TrLongMat = getTrialLongMat(TrInfo,TrCondMat)
    # long form matrix of the firing rates by trial and position that matches
    # TrLongMat
    TrFRData = getTrialLongDatMat(TrInfo)

    ################################################################################
    # save data
    TrInfo_Fn = sessionPaths['Analyses'] / 'TrInfo.pkl'
    with TrInfo_Fn.open(mode='wb') as f:
        pkl.dump(TrInfo,f,pkl.HIGHEST_PROTOCOL)

    TrCondMat_Fn = sessionPaths['Analyses'] / 'TrialCondMat.pkl'
    with TrCondMat_Fn.open(mode='wb') as f:
        pkl.dump(TrCondMat,f,pkl.HIGHEST_PROTOCOL)

def getTrInfo(PosDat):
    # get trial durations and samples
    TrialVec = PosDat['EventDat']['TrID']
    nTr=TrialVec.max()

    startIDs = np.zeros(nTr,dtype=int)
    endIDs = np.zeros(nTr,dtype=int)
    for tr in np.arange(nTr):
        trIDs = np.where(TrialVec==(tr+1))[0]
        startIDs[tr]=trIDs[0]
        endIDs[tr] = trIDs[-1]

    TrialDurs = endIDs-startIDs

    OffTrialDurs=np.concatenate((startIDs[1:],[len(PosDat['t'])]))-endIDs
    OffTrialVec = np.full_like(TrialVec,0)

    for tr in np.arange(nTr):
        idx = np.arange(endIDs[tr],endIDs[tr]+OffTrialDurs[tr])
        OffTrialVec[idx]=tr+1

    # Pre allocated Trial Info structure.
    TrInfo = {'All':{'Trials':[],'Co':[],'InCo':[]},'L':{'Trials':[],'Co':[],'InCo':[]},
                 'R':{'Trials':[],'Co':[],'InCo':[]},'BadTr':[],'Cues':np.full(nTr,''),'Desc':np.full(nTr,''),
                 'DurThr':45,'TrDurs':TrialDurs,
                 'TrialVec':TrialVec,'TrStSamp':startIDs,'TrEnSamp':endIDs,'TrSeq':{'Pos':{},'Samp':{},'Dur':{}},
                 'OffTrStSamp':endIDs,'OffTrEnSamp':endIDs+OffTrialDurs,'OffTrDurs':OffTrialDurs,
                 'OffTrialVec':OffTrialVec, 'OffTrSeq':{'Pos':{},'Samp':{},'Dur':{}},
                 'ValidSeqTrials':[],'ValidSeqOffTrials':[],'ValidSeqTrID':[],'ValidSeqOffTrID':[],
                 'ValidSeqNames':ValidTraj,'ValidSeqOffNames':ValidOffTraj}

    TrInfo['All']['Trials']=np.arange(nTr)+1
    #get separate trials and allocate by correct/incorrect
    for tr in TrInfo['All']['Trials']:
        idx= TrialVec==tr
        for s in ['L','R']:
            c = PosDat['EventDat']['C'+s][idx]
            d = PosDat['EventDat'][s+'Ds'][idx]
            if np.mean(d)>0.5: # descicion
                TrInfo['Desc'][tr-1]=s
            if np.mean(c)>0.5: # cue
                TrInfo[s]['Trials'].append(tr)
                TrInfo['Cues'][tr-1]=s
                if np.mean(d&c)>0.5: # correct descicion
                    TrInfo[s]['Co'].append(tr)
                else:
                    TrInfo[s]['InCo'].append(tr)
    assert set(TrInfo['R']['Trials']) & set(TrInfo['L']['Trials']) == set(), 'Trial classified as both left and right.'
    assert len(TrInfo['Cues']) ==len(TrInfo['Desc']), 'Number of trials mismatch'
    assert len(TrInfo['Cues']) ==nTr, 'Number of trials mismatch'

    for trC in ['Co', 'InCo']:
        TrInfo['All'][trC] = np.sort(TrInfo['L'][trC]+TrInfo['R'][trC])

    for i in ['All','L','R']:
        for j in ['Trials','Co','InCo']:
            TrInfo[i]['n'+j]=len(TrInfo[i][j])
            TrInfo[i][j]=np.array(TrInfo[i][j])

    # determine if the trials are too long to be included.
    TrInfo['BadTr'] = np.where(TrInfo['TrDurs']*PosDat['step']>TrInfo['DurThr'])[0]

    # get positions for each trial
    for tr in TrInfo['All']['Trials']:

        idx = TrInfo['TrialVec']==tr
        sID = TrInfo['TrStSamp'][tr-1]
        eID = TrInfo['TrEnSamp'][tr-1]

        p,s,d=getPosSequence(PosDat['PosZones'][idx],sID,eID)
        #p,s,d=getPosSequence(p4[idx],sID,eID)

        TrInfo['TrSeq']['Pos'][tr]=p
        TrInfo['TrSeq']['Samp'][tr]=s
        TrInfo['TrSeq']['Dur'][tr]=d

        idx = TrInfo['OffTrialVec']==tr
        sID = TrInfo['OffTrStSamp'][tr-1]
        eID = TrInfo['OffTrEnSamp'][tr-1]

        p,s,d=getPosSequence(PosDat['PosZones'][idx],sID,eID)

        TrInfo['OffTrSeq']['Pos'][tr]=p
        TrInfo['OffTrSeq']['Samp'][tr]=s
        TrInfo['OffTrSeq']['Dur'][tr]=d

    # determine if the sequence of positions are valid for each trial
    TrSeqs = {}
    vTr = []
    OffTrSeqs = {}
    vOTr = []
    for tr in TrInfo['All']['Trials']:
        seq = [TMF.Zones[a] for a in TrInfo['TrSeq']['Pos'][tr]]
        match = 0
        for vSeqN, vSeq in ValidTraj.items():
            if cmp(seq,vSeq)==0:
                match = 1
                vTr.append(tr)
                TrSeqs[tr]=vSeqN
                break
        if match==0:
            TrSeqs[tr]=[]

        seq =  [TMF.Zones[a] for a in TrInfo['OffTrSeq']['Pos'][tr]]
        match = 0
        for vSeqN, vSeq in ValidOffTraj.items():
            if cmp(seq,vSeq)==0:
                match = 1
                vOTr.append(tr)
                OffTrSeqs[tr]=vSeqN
                break
        if match==0:
            OffTrSeqs[tr]=[]

    TrInfo['ValidSeqTrials'] = vTr
    TrInfo['ValidSeqOffTrials'] = vOTr
    TrInfo['ValidSeqTrID'] = TrSeqs
    TrInfo['ValidSeqOffTrID'] = OffTrSeqs
    return TrInfo

def getTrCondMat(TrInfo):
    nTr=len(TrInfo['All']['Trials'])
    conds = ['Cues','Desc','Co','Traj','OTraj','Dur','Good','Length','OLength']
    TrCondMat = pd.DataFrame(np.full((nTr,len(conds)),np.nan),index=TrInfo['All']['Trials'],columns=conds)

    TrCondMat['Cues'] = TrInfo['Cues']
    TrCondMat['Desc'] = TrInfo['Desc']
    TrCondMat['Dur'] = TrInfo['TrDurs']

    TrCondMat['Co'].loc[TrInfo['All']['Co']]='Co'
    TrCondMat['Co'].loc[TrInfo['All']['InCo']]='InCo'

    vseq=TrInfo['ValidSeqTrials']
    TrCondMat['Traj'].loc[vseq]=[TrInfo['ValidSeqTrID'][s] for s in vseq]

    vseq=TrInfo['ValidSeqOffTrials']
    TrCondMat['OTraj'].loc[vseq]=[TrInfo['ValidSeqOffTrID'][s] for s in vseq]

    TrCondMat['Good'] = (~TrCondMat['Traj'].isnull()) & (TrialDurs*PosDat['step']<TrInfo['DurThr'])

    x=np.full(nTr,'')
    for k,v in TrInfo['ValidSeqTrID'].items():
        if len(v)>0:
            x[k-1]=v[2]
    TrCondMat['Length']= x

    x=np.full(nTr,'')
    for k,v in TrInfo['ValidSeqOffTrID'].items():
        if len(v)>0:
            x[k-1]=v[2]
    TrCondMat['OLength']= x
    return TrCondMat

def getTrLongMat(TrInfo,TrCondMat):
    nMaxPos = 11
    nDatStack = 3 # Out, In, O-I
    nTr =len(TrInfo['All']['Trials'])

    Cols = ['trID','Pos','IO','Cue','Desc','Traj','Loc','OTraj', 'Goal','ioMatch','Co','Valid']
    nCols = len(Cols)
    TrLongMat = pd.DataFrame(np.full((nTr*nMaxPos*nDatStack,nCols),np.nan),columns=Cols)

    TrLongMat['trID'] = np.tile(np.tile(TrInfo['All']['Trials'],nMaxPos),nDatStack)
    TrLongMat['Pos'] = np.tile(np.repeat(np.arange(nMaxPos),nTr),nDatStack)
    TrLongMat['IO'] = np.repeat(['Out','In','O_I'],nTr*nMaxPos)

    TrLongMat['Traj'] = np.tile(np.tile(TrCondMat['Traj'],nMaxPos),nDatStack)
    TrLongMat['OTraj'] = np.tile(np.tile(TrCondMat['OTraj'],nMaxPos),nDatStack)
    TrLongMat['Co'] = np.tile(np.tile(TrCondMat['Co'],nMaxPos),nDatStack)
    TrLongMat['Cue'] = np.tile(np.tile(TrCondMat['Cues'],nMaxPos),nDatStack)
    TrLongMat['Desc'] = np.tile(np.tile(TrCondMat['Desc'],nMaxPos),nDatStack)
    TrLongMat['ioMatch'] = [traj==otraj for traj,otraj in zip(TrLongMat['Traj'],TrLongMat['OTraj'])]
    TrLongMat['Goal'] = [traj[3] if traj==traj else '' for traj in TrLongMat['Traj']]
    TrLongMat['Len'] = [traj[2] if traj==traj else '' for traj in TrLongMat['Traj']]
    TrLongMat['OLen'] = [traj[2] if traj==traj else '' for traj in TrLongMat['OTraj']]

    # get true location in each trials sequence 'Loc'
    # note that 'Pos' is a numerical indicator of the order in a sequence
    outTrSeq = pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'])
    inTrSeq = pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'])
    oiTrSeq = pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'])

    for tr in TrInfo['All']['Trials']:
        traj = TrInfo['ValidSeqTrID'][tr]
        if traj in ValidTrajNames:
            seq =  TrInfo['ValidSeqNames'][traj]
            if len(seq)==nMaxPos:
                outTrSeq.loc[tr]=seq
            else:
                outTrSeq.loc[tr]= seq + [np.nan]*4
        else:
            outTrSeq.loc[tr] = [np.nan]*nMaxPos

        otraj = TrInfo['ValidSeqOffTrID'][tr]
        if otraj in ValidTrajNames:
            oseq =  TrInfo['ValidSeqOffNames'][otraj]
            if len(oseq)==nMaxPos:
                inTrSeq.loc[tr]=oseq
            else:
                inTrSeq.loc[tr]=[np.nan]*4+oseq
        else:
            inTrSeq.loc[tr]=[np.nan]*nMaxPos

        if (traj in ValidTrajNames) and (otraj in ValidTrajNames):
            if traj==otraj:
                if len(seq)==nMaxPos:
                    oiTrSeq.loc[tr]=seq
                else:
                    oiTrSeq.loc[tr] = seq + [np.nan]*4
            elif traj[2]=='L' and otraj[2]=='S':
                oiTrSeq.loc[tr] = seq[:4]+[np.nan]+seq[9:]+[np.nan]*4
            elif traj[2]=='S' and otraj[2]=='L':
                oiTrSeq.loc[tr] = seq[:4]+[np.nan]+seq[5:]+[np.nan]*4
        else:
            oiTrSeq.loc[tr] = [np.nan]*nMaxPos

    TrLongMat['Loc'] = pd.concat([pd.concat([outTrSeq.melt(value_name='Loc')['Loc'],inTrSeq.melt(value_name='Loc')['Loc']]),oiTrSeq.melt(value_name='Loc')['Loc']]).values
    TrLongMat['Valid'] = ~TrLongMat['Loc'].isnull()
    TrLongMat['EvenTrial'] = TrLongMat['trID']%2==0
    return TrLongMat

def getTrLongDatMat(TrInfo):
    # working version of long DF trialxPos matrix. this takes either short or long trajectories in the outbound but only short trajectories inbound
    nMaxPos = 11
    nMinPos = 7
    nTr =len(TrInfo['All']['Trials'])

    nCells = cell_FR.shape[0]
    nMua = mua_FR.shape[0]
    nTotalUnits = nCells+nMua
    nUnits = {'cell':nCells,'mua':nMua}

    TrZn={'cell':[],'mua':[]}
    OTrZn={'cell':[],'mua':[]}

    TrZn['cell'],OTrZn['cell'] = getFR_TrZone(TrInfo,cell_FR)
    TrZn['mua'],OTrZn['mua'] = getFR_TrZone(TrInfo,mua_FR)

    cellCols = ['cell_'+str(i) for i in np.arange(nCells)]
    muaCols = ['mua_'+str(i) for i in np.arange(nMua)]
    unitCols = {'cell':cellCols,'mua':muaCols}
    allUnits = cellCols+muaCols

    mu = {'cell':np.mean(cell_FR,1),'mua':np.mean(mua_FR,1)}
    sig = {'cell':np.std(cell_FR,1),'mua':np.std(mua_FR,1)}

    Out=pd.DataFrame(np.full((nTr*nMaxPos,nTotalUnits),np.nan),columns=allUnits)
    In=pd.DataFrame(np.full((nTr*nMaxPos,nTotalUnits),np.nan),columns=allUnits)
    O_I=pd.DataFrame(np.full((nTr*nMaxPos,nTotalUnits),np.nan),columns=allUnits)

    Out = Out.assign(trID = np.tile(TrInfo['All']['Trials'],nMaxPos))
    In = In.assign(trID = np.tile(TrInfo['All']['Trials'],nMaxPos))
    O_I = O_I.assign(trID = np.tile(TrInfo['All']['Trials'],nMaxPos))

    Out = Out.assign(Pos = np.repeat(np.arange(nMaxPos),nTr))
    In = In.assign(Pos = np.repeat(np.arange(nMaxPos),nTr))
    O_I = O_I.assign(Pos = np.repeat(np.arange(nMaxPos),nTr))

    Out = Out.assign(IO = ['Out']*(nTr*nMaxPos))
    In = In.assign(IO = ['In']*(nTr*nMaxPos))
    O_I = O_I.assign(IO = ['O_I']*(nTr*nMaxPos))

    for ut in ['cell','mua']:
        for cell in np.arange(nUnits[ut]):
            X=pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'],columns=np.arange(nMaxPos))
            Y=pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'],columns=np.arange(nMaxPos))
            Z=pd.DataFrame(np.full((nTr,nMaxPos),np.nan),index=TrInfo['All']['Trials'],columns=np.arange(nMaxPos))

            m = mu[ut][cell]
            s = sig[ut][cell]

            for tr in TrInfo['All']['Trials']:
                traj = TrInfo['ValidSeqTrID'][tr]
                if traj in ValidTrajNames:
                    if traj[2]=='S':
                        X.loc[tr][0:nMinPos] = zscore(TrZn[ut][tr][cell],m,s)
                    else:
                        X.loc[tr] = zscore(TrZn[ut][tr][cell],m,s)

                otraj = TrInfo['ValidSeqOffTrID'][tr]
                if otraj in ValidTrajNames:
                    if otraj[2]=='S':
                        Y.loc[tr][4:] = zscore(OTrZn[ut][tr][cell],m,s)
                    else:
                        Y.loc[tr] = zscore(OTrZn[ut][tr][cell],m,s)
                if (traj in ValidTrajNames) and (otraj in ValidTrajNames):
                    if traj==otraj:
                        Z.loc[tr] = X.loc[tr].values-Y.loc[tr][::-1].values
                    elif traj[2]=='L' and otraj[2]=='S': # ambigous interserction position, skipping that computation
                        Z.loc[tr][[0,1,2,3]] = X.loc[tr][[0,1,2,3]].values-Y.loc[tr][[10,9,8,7]].values
                        Z.loc[tr][[5,6]] = X.loc[tr][[9,10]].values-Y.loc[tr][[5,4]].values
                    elif traj[2]=='S' and otraj[2]=='L':
                        Z.loc[tr][[0,1,2,3]] = X.loc[tr][[0,1,2,3]].values-Y.loc[tr][[10,9,8,7]].values
                        Z.loc[tr][[5,6]] = X.loc[tr][[5,6]].values-Y.loc[tr][[1,0]].values

            Out[unitCols[ut][cell]]=X.melt(value_name='zFR')['zFR']
            In[unitCols[ut][cell]]=Y.melt(value_name='zFR')['zFR']
            O_I[unitCols[ut][cell]]=Z.melt(value_name='zFR')['zFR']

    Data = pd.DataFrame()
    Data =  pd.concat([Data,Out])
    Data =  pd.concat([Data,In])
    Data =  pd.concat([Data,O_I])
    Data = Data.reset_index()
    return Data

def fitTrModels(TrLongMat,TrFRData,cell_FR,mua_FR):
    all_params = ['Loc:IO','Loc','IO','Cue','Desc','Co']
    param_set = getParamSet()
    nModels = len(param_set)
    R2thr = 0.2

    nTr =len(TrInfo['All']['Trials'])
    nCells = cell_FR.shape[0]
    nMua = mua_FR.shape[0]
    nTotalUnits = nCells+nMua
    nUnits = {'cell':nCells,'mua':nMua}
    cellCols = ['cell_'+str(i) for i in np.arange(nCells)]
    muaCols = ['mua_'+str(i) for i in np.arange(nMua)]
    unitCols = {'cell':cellCols,'mua':muaCols}

    perfCols = ['FullMod_tR2','modelNum','trainR2','AICc','testR2']
    Cols = ['ut']+perfCols+ all_params
    nCols = len(Cols)
    LM_Dat = pd.DataFrame(np.full((nTotalUnits,nCols),np.nan),columns=Cols)
    LM_Dat.loc[:,'ut'] = ['cell']*nCells+['mua']*nMua

    datSubset = ~(TrLongMat['IO']=='O_I') & (TrLongMat['Valid'])
    dat = []
    dat = TrLongMat[datSubset].copy()
    dat['trID'] = dat['trID'].astype('category')
    dat = dat.reset_index()
    N = dat.shape[0]
    dat['zFR'] = np.zeros(N)

    with Parallel(n_jobs=16) as parallel:
        cnt=0
        for ut in ['cell','mua']:
            for cell in np.arange(nUnits[ut]):
                print('\nAnalyzing {} {}'.format(ut,cell))

                dat.loc[:,'zFR'] = Data.loc[datSubset,unitCols[ut][cell]].values

                t1 = time.time()
                tR2 = getModel_testR2(dat,params=all_params)
                t2 = time.time()
                LM_Dat.loc[cell,'FullMod_tR2'] = tR2

                print('Full Model Test Set Fit completed. Time = {}'.format(t2-t1))
                if tR2>=R2thr:
                    print('Full Model passed the threshold, looking for optimal submodel.')
                    r = parallel(delayed(getModelPerf)(dat,params=params) for params in param_set.values())
                    trainR2,trainAICc,_ = zip(*r)

                    t3 = time.time()
                    print('\n\nFitting Completed for cell {0}, total time = {1:0.3f}s'.format(cell,t3-t1))
                    selMod = np.argmin(trainAICc)

                    selMod_tR2 = getModel_testR2(dat,params=param_set[selMod])
                    print('Selected Model = {}, AICc = {}, testR2 = {} '.format(selMod,trainAICc[selMod],selMod_tR2))

                    LM_Dat.loc[cnt,'modelNum']=selMod
                    LM_Dat.loc[cnt,'trainR2']=trainR2[selMod]
                    LM_Dat.loc[cnt,'AICc'] = trainAICc[selMod]
                    LM_Dat.loc[cnt,'testR2'] = selMod_tR2

                    temp = r[selMod][2].wald_test_terms()
                    LM_Dat.loc[cnt][params] = np.sqrt(temp.summary_frame()['chi2'][params])
                cnt+=1
    return d

def AICc(model):
    n = model.nobs
    llf = model.llf
    k = len(model.params)
    AIC = 2*(k-llf)
    c = 2*k*(k+1)/(n-k-1)
    return AIC+c

def R2(x,y):
    return (np.corrcoef(x,y)**2)[0,1]

def aR2(model,y,fit=[]):
    if fit==[]:
        fit = model.fittedvalues
    r2 = R2(fit,y)
    n = model.nobs
    p = len(model.params)-1
    aR2 = 1-(1-r2)*(n-1)/(n-p-1)
    return aR2

def getParamSet():
    '''
    Returns a dictionary of parameter sets for modeling.
    '''
    params = ['Loc','IO','Cue','Desc','Co']
    combs = []

    for i in np.arange(1, len(params)+1):
        combs+= [list(x) for x in combinations(params, i)]
    param_set = {}
    cnt=0
    for c in combs:
        param_set[cnt] = c
        cnt+=1

    for c in combs:
        if ('IO' in c) and ('Loc' in c):
            param_set[cnt] = ['Loc:IO']+c
            cnt+=1
    return param_set

def getModel_testR2(dat,formula='',params=[],mixedlm=True, verbose=False):
    '''
    Obtains the test R2 based on even/odd splits of the data
    '''

    if len(params)>0 and len(formula)==0:
        formula = getFormula(params)
    else:
        print('No Method of selecting parameters provided.')
        return np.nan,np.nan,[]
    print('\nComputing mixedlm with formula: {}'.format(formula))

    dat_even = dat[dat['EvenTrial']==True]
    dat_odd = dat[dat['EvenTrial']==False]

    if mixedlm:
        md_even = smf.mixedlm(formula, data=dat_even,groups=dat_even["trID"])
    else:
        md_even = smf.ols(formula + 'trID', data=dat_even)
    mdf_even = md_even.fit()
    pred_odd = mdf_even.predict(dat_odd)


    if mixedlm:
        md_odd = smf.mixedlm(formula, data=dat_odd,groups=dat_odd["trID"])
    else:
        md_odd = smf.ols(formula + 'trID', data=md_odd)

    mdf_odd = md_odd.fit()
    pred_even = mdf_odd.predict(dat_even)

    if verbose:
        print('\nPerformance Train-Even:Test-Odd')
        print("Train_aR2 = {0:.3f}".format(aR2(mdf_even,dat_even['zFR'])))
        print("Model_AICc = {0:.3f}".format(AICc(mdf_even)))
        print("Test_R2 = {0:.3f}".format(R2(pred_odd,dat_odd['zFR'])))
        print('\nPerformance Train-Odd:Test-Even')
        print("Train_aR2 = {0:.3f}".format(aR2(mdf_odd,dat_odd['zFR'])))
        print("Model_AICc = {0:.3f}".format(AICc(mdf_odd)))
        print("Test_R2 = {0:.3f}".format(R2(pred_even,dat_even['zFR'])))

    dat['Pred']=np.zeros(dat.shape[0])
    dat.loc[dat['EvenTrial']==True,'Pred']=pred_even
    dat.loc[dat['EvenTrial']==False,'Pred']=pred_odd

    r2 = R2(dat['zFR'],dat['Pred'])
    print('\nOverall test R2: {0:.3f}'.format(r2))
    return r2

def getFormula(params):
    formula = 'zFR ~ '
    nP = len(params)
    cnt=1
    for i in params:
        formula += i
        if cnt<nP:
            formula +='+'
        cnt+=1
    return formula

def getModelPerf(dat,formula='',params=[],mixedlm=True):
    '''
    Obtains the train adjusted R2, and AIC for data.
    returns aR2, AIC, and the fitted model.
    '''
    print('\nComputing mixedlm with formula: {}'.format(formula))

    if len(params)>0 and len(formula)==0:
        formula = getFormula(params)
    else:
        print('No Method of selecting parameters provided.')
        return np.nan,np.nan,[]

    if mixedlm:
        md = smf.mixedlm(formula, data=dat, groups=dat["trID"])
    else:
        md = smf.ols(formula + '+trID', data=dat)

    mdf = md.fit()
    print('\n Model Performance:')
    train_aR2 = aR2(mdf,dat['zFR'])
    print("Train_aR2 = {0:.3f}".format(train_aR2))
    aic = AICc(mdf)
    print("Model_AICc = {0:.3f}".format(aic))

    return train_aR2, aic, mdf

########################################
sns.set()
sns.set(style="whitegrid",context='notebook',font_scale=1.5,rc={
    'axes.spines.bottom': False,
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.edgecolor':'0.5'})
#,rc={'grid.color':'0.1','axes.labelcolor':'1'})
#sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
pal = sns.xkcd_palette(['green','purple'])

f,ax = plt.subplots(1,3, figsize=(15,4))
yPos = 0.04
w = 0.25
ratio = 7.5/11.5
hsp = 0.05
W = [w,w*ratio,w*ratio]
xPos = [hsp,2*hsp+W[0],3*hsp+W[1]+W[0]]
H = 0.85
xlims = [[-0.25,10.25],[-0.25,6.25],[-0.25,6.25]]
for i in np.arange(3):
    ax[i].set_position([xPos[i],yPos,W[i],H])
    ax[i].set_xlim(xlims[i])
    ax[i].axhline(color=[0.7,0.7,0.7],alpha=0.7)

xPosLabels = ['Home','SegA','Center','SegBE','Int','CDFG','Goals','CDFG','Int','CDFG','Goals']

xPosLabels2 = ['Home','SegA','Center','SegBE','Int','CDFG','Goals'][::-1]
with sns.color_palette(pal):
    subset = (cellDat['IO']=='Out') & (cellDat['Co']=='Co')
    ax[0] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',data=cellDat[subset],ax=ax[0],legend=False,hue_order=['L','R'])
    ax[0].set_xticks(np.arange(11))

    ax[0].tick_params(axis='x', rotation=60)
    ax[0].set_xticklabels(xPosLabels)
    ax[0].set_xlabel('')
    ax[0].set_title('Out')

    subset = (cellDat['IO']=='In') & (cellDat['Co']=='Co')
    ax[1] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',data=cellDat[subset],ax=ax[1],legend=False,hue_order=['L','R'])
    ax[1].set_xticks(np.arange(7))
    ax[1].tick_params(axis='x', rotation=60)
    ax[1].set_xticklabels(xPosLabels2)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')
    ax[1].set_title('In')

    subset = (cellDat['IO']=='O-I') & (cellDat['Co']=='Co')
    ax[2] = sns.lineplot(x='Pos',y='zFR',hue='Cue',style='Goal',data=cellDat[subset],ax=ax[2],legend='brief',hue_order=['L','R'])
    ax[2].set_xticks(np.arange(7))
    ax[2].tick_params(axis='x', rotation=60)
    ax[2].set_xticklabels(xPosLabels2)
    ax[2].set_xlabel('')
    ax[2].set_ylabel('')
    ax[2].set_title('O-I')

    lims = np.zeros((3,2))
    for i in np.arange(3):
        lims[i]=np.array(ax[i].get_ylim())
    minY = np.floor(np.min(lims[:,0])*20)/20
    maxY = np.ceil(np.max(lims[:,1]*20))/20

    for i in np.arange(3):
        ax[i].set_ylim([minY,maxY])
    l =ax[2].get_legend()
    #l.set_bbox_to_anchor=(0, 0,0.3,1)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.,frameon=False)
    l.set_frame_on(False)
    #l.set_visible(False)

##################
sns.set(style="whitegrid",font_scale=1.5)
pal = sns.xkcd_palette(['spring green','light purple'])
subset = cellDat['Co']=='Co'
dat = cellDat[subset].groupby(['trID','IO','Cue','Desc']).mean()
dat = dat.reset_index()
with sns.color_palette(pal):
    ax=sns.violinplot(y='zFR',x='IO',hue='Desc',data=dat,split=True,scale='count',inner='quartile',hue_order=['L','R'],saturation=0.7)
pal = sns.xkcd_palette(['emerald green','medium purple'])
with sns.color_palette(pal):
    ax=sns.swarmplot(y='zFR',x='IO',hue='Desc',data=dat,dodge=True,hue_order=['L','R'],alpha=0.7,edgecolor='gray')
l=ax.get_legend()
l.set_visible(False)
ax.set_xlabel('Direction')

###############
sns.set(style="whitegrid",font_scale=1.5)
pal = sns.xkcd_palette(['spring green','light purple'])
subset= cellDat['IO']=='Out'
dat = cellDat[subset].groupby(['trID','Cue','Co','Desc']).mean()
dat = dat.reset_index()

with sns.color_palette(pal):
    ax=sns.violinplot(y='zFR',x='Desc',hue='Cue',data=dat,split=False,scale='width',inner='quartile',order=['L','R'],hue_order=['L','R'],saturation=0.7)
pal = sns.xkcd_palette(['emerald green','medium purple'])
with sns.color_palette(pal):
    ax=sns.swarmplot(y='zFR',x='Desc',hue='Cue',data=dat,dodge=True,order=['L','R'],hue_order=['L','R'],alpha=0.7,edgecolor='gray')
l=ax.get_legend()
l.set_visible(False)
#plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.,frameon=False,title='Cue')
ax.set_xlabel('Decision')

########
sns.set(style="whitegrid",font_scale=1.5)
pal = sns.xkcd_palette(['spring green','light purple'])
subset= cellDat['IO']=='Out'
dat = cellDat[subset].groupby(['trID','Cue','Co','Desc']).mean()
dat = dat.reset_index()

with sns.color_palette(pal):
    ax=sns.violinplot(y='zFR',x='Desc',hue='Cue',data=dat,split=False,scale='width',inner='quartile',order=['L','R'],hue_order=['L','R'],saturation=0.7)
pal = sns.xkcd_palette(['emerald green','medium purple'])
with sns.color_palette(pal):
    ax=sns.swarmplot(y='zFR',x='Desc',hue='Cue',data=dat,dodge=True,order=['L','R'],hue_order=['L','R'],alpha=0.7,edgecolor='gray')
l=ax.get_legend()
l.set_visible(False)
#plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0.,frameon=False,title='Cue')
ax.set_xlabel('Decision')
