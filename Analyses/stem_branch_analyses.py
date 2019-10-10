import numpy as np
import pandas as pd
from scipy import stats
import sys, os, time, json
from pathlib import Path
import pickle as pkl

sys.path.append('../PreProcessing/')
sys.path.append('../Lib/')
sys.path.append('../Analyses/')

import sklearn.linear_model as lm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score as bac
from joblib import Parallel, delayed

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.text import Text
import seaborn as sns

import analyses_table as AT
import TreeMazeFunctions as TMF

sns.set(style="whitegrid",font_scale=1,rc={
    'axes.spines.bottom': False,
'axes.spines.left': False,
'axes.spines.right': False,
'axes.spines.top': False,
'axes.edgecolor':'0.5'})

def main(sePaths, doPlots=False, overwrite = False):
    try:
        dat = AT.loadSessionData(sePaths)
        nUnits = dat['TrialModelFits'].shape[0]

        # univariate analyses.
        fn = sePaths['CueDesc_SegUniRes']
        if ( (not fn.exists()) or overwrite):
            CueDescFR_Dat, all_dat_spl = CueDesc_SegUniAnalysis(dat)
            CueDescFR_Dat.to_csv(sePaths['CueDesc_SegUniRes'])

            if doPlots:
                plotCueVDes(CueDescFR_Dat,sePaths)
                plotUnitRvL(CueDescFR_Dat,all_dat_spl,sePaths)
        else:
            CueDescFR_Dat = pd.read_csv(fn)

        # decododer analyses
        fn = sePaths['CueDesc_SegDecRes']
        if ((not fn.exists()) or overwrite):
            singCellDec,singCellDecSummary, popDec = CueDesc_SegDecAnalysis(dat)
            singCellDec['se'] = sePaths['session']
            singCellDecSummary['se'] = sePaths['session']
            popDec['se'] = sePaths['session']
            singCellDec.to_csv(fn)
            singCellDecSummary.to_csv(sePaths['CueDesc_SegDecSumRes'])
            popDec.to_csv(sePaths['PopCueDesc_SegDecSumRes'])

            if doPlots:
                f,_ = plotMultipleDecoderResults(singCellDecSummary)
                fn = sePaths['CueDescPlots'] / ('DecResByUnit.jpeg')
                f.savefig(str(fn),dpi=150, bbox_inches='tight',pad_inches=0.2)
                plt.close(f)

                f,_ = plotMultipleDecoderResults(popDec)
                fn = sePaths['CueDescPlots'] / ('PopDecRes.jpeg')
                f.savefig(str(fn),dpi=150, bbox_inches='tight',pad_inches=0.2)
                plt.close(f)

                for unit in np.arange(nUnits):
                    f,_ = plotMultipleDecoderResults(singCellDec[(singCellDec['unit']==unit)])
                    fn = sePaths['CueDescPlots'] / ('DecRes_UnitID-{}.jpeg'.format(unitNum) )
                    f.savefig(str(fn),dpi=150, bbox_inches='tight',pad_inches=0.2)
                    plt.close(f)
        else:
            singCellDec = pd.read_csv(fn)
            singCellDecSummary = pd.read_csv(sePaths['CueDesc_SegDecSumRes'])
            popDec = pd.read_csv(sePaths['PopCueDesc_SegDecSumRes'])

        return CueDescFR_Dat, singCellDec,singCellDecSummary, popDec
    except:
        print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
        return [],[],[],[]

def CueDesc_SegUniAnalysis(dat):
    trDat = dat['TrialLongMat']
    trConds = dat['TrialConds']
    nCells = len(dat['ids']['cells'])
    nMua = len(dat['ids']['muas'])
    nUnits = nCells+nMua

    # fixed variables (don't change with cell)
    locs = TMF.ZonesNames
    Trials = trConds[trConds['Good']].index.values
    nTrials = len(Trials)

    FeatIDs = {'A':[1],'Stem':[0,1,2],'Arm': [3,4]}
    Segs =  FeatIDs.keys()
    HA = ['Home','SegA']
    Stem = ['Home','SegA','Center']
    L_Arm = ['SegE', 'I2', 'SegF', 'G3', 'SegG', 'G4']
    R_Arm = ['SegB', 'I1', 'SegC', 'G1', 'SegD', 'G2']

    # variable to be stored
    #uni_LvR_Analyses = {'Stats':{'Cue':{},'Desc':{},'Cue_Desc':{}},'Mean':{'Cue':{},'Desc':{},'Cue_Desc':{}},'SD':{'Cue':{},'Desc':{},'Cue_Desc':{}} }
    uni_LvR_Analyses = {'Cue':{'Stats':{},'Mean':{},'SD':{}},'Desc':{'Stats':{},'Mean':{},'SD':{}},'Cue_Desc':{'Stats':{},'Mean':{},'SD':{}}}

    Conds = ['Cue','Desc','Cue_Desc']
    dat_meas = ['Stats','Mean','SD']

    all_dat_spl = {} # only used for plotting as it has overlapping data points; not necessary to store it.

    for unitNum in np.arange(nUnits):

        # splits of data per cell
        dat_splits = {}
        for k in ['Cue','Desc']:
            dat_splits[k] = {}
            for kk in FeatIDs.keys():
                dat_splits[k][kk] = {}
        dat_splits['Cue_Desc'] = {'Co_Arm':{},'L_Arm':{},'R_Arm':{}}


        if unitNum==0:
            for k in Conds:
                for ii in dat_meas:
                    if ii=='Stats':
                        for jj in ['T','P','S']:
                            uni_LvR_Analyses[k][ii][jj] = pd.DataFrame(np.zeros((nUnits,3)),columns=dat_splits[k].keys())
                    else:
                        for jj in ['L','R']:
                            uni_LvR_Analyses[k][ii][jj] = pd.DataFrame(np.zeros((nUnits,3)),columns=dat_splits[k].keys())

        if unitNum<nCells:
            tt = dat['ids']['cells'][str(unitNum)][0]
            cl = dat['ids']['cells'][str(unitNum)][1]
            fr = dat['TrialFRLongMat']['cell_'+str(unitNum)]
            tR2 = dat['TrialModelFits']['testR2'][unitNum]
            selMod = dat['TrialModelFits']['selMod'][unitNum]
        else:
            muaID = unitNum-nCells
            tt = dat['ids']['muas'][str(muaID)][0]
            cl = dat['ids']['muas'][str(muaID)][1]
            fr = dat['TrialFRLongMat']['mua_'+str(muaID)]
            tR2 = dat['TrialModelFits']['testR2'][unitNum]
            selMod = dat['TrialModelFits']['selMod'][unitNum]

        # get mean fr per trial per partition
        mPartFRDat = pd.DataFrame(np.zeros((nTrials,3)),columns=FeatIDs)
        cue = trConds.loc[Trials,'Cues'].values
        desc = trConds.loc[Trials,'Desc'].values
        cnt =0
        for tr in Trials:
            subset = (trDat['trID']==tr) & (trDat['IO']=='Out')
            for k,v in FeatIDs.items():
                mPartFRDat.loc[cnt,k]=np.nanmean(fr[subset].values[v])
            cnt+=1

        # univariate cue and desciscion tests by maze part
        LvR = {}
        l = {}
        r = {}

        # First & Second analyses: Cue/Desc
        k = 'Cue'
        l[k] = cue=='L'
        r[k] = cue=='R'

        k = 'Desc'
        l[k]=desc=='L'
        r[k]=desc=='R'

        for k in ['Cue','Desc']:
            LvR[k] = pd.DataFrame(np.zeros((3,3)),index=Segs,columns=['T','P','S'])
            for kk in Segs:
                lfr = mPartFRDat[kk][l[k]]
                rfr = mPartFRDat[kk][r[k]]
                temp = stats.ttest_ind(lfr,rfr)
                LvR[k].loc[kk,'T'] = temp[0]
                LvR[k].loc[kk,'P'] = temp[1]
                dat_splits[k][kk]['l'] = lfr.values
                dat_splits[k][kk]['r'] = rfr.values
            LvR[k]['S'] = getSigLevel(LvR[k]['P'])

        # thir analysis: Correct v Incorrect by L/R arm
        k = 'Cue_Desc'
        LvR[k] = pd.DataFrame(np.zeros((3,3)),index=['Co_Arm','L_Arm','R_Arm'],columns=['T','P','S'])

        l = {}
        r = {}

        kk = 'Co_Arm'
        l[kk] = mPartFRDat['Arm'][(cue=='L')&(desc=='L')]
        r[kk] = mPartFRDat['Arm'][(cue=='R')&(desc=='R')]

        kk = 'L_Arm'
        l[kk]=mPartFRDat['Arm'][(desc=='L')&(cue=='L')]
        r[kk]=mPartFRDat['Arm'][(desc=='L')&(cue=='R')]

        kk = 'R_Arm'
        l[kk]=mPartFRDat['Arm'][(desc=='R')&(cue=='L')]
        r[kk]=mPartFRDat['Arm'][(desc=='R')&(cue=='R')]

        for kk in ['Co_Arm','L_Arm','R_Arm']:
            temp = stats.ttest_ind(l[kk],r[kk])
            LvR[k].loc[kk,'T'] = temp[0]
            LvR[k].loc[kk,'P'] = temp[1]
            dat_splits[k][kk]['l'] = l[kk].values
            dat_splits[k][kk]['r'] = r[kk].values

        LvR[k]['S'] = getSigLevel(LvR[k]['P'])

        # aggreagate results.
        mlr = {}
        slr = {}
        for k,v in dat_splits.items():
            mlr[k] = pd.DataFrame(np.zeros((3,2)),index=v.keys(),columns=['L','R'])
            slr[k] = pd.DataFrame(np.zeros((3,2)),index=v.keys(),columns=['L','R'])
            cnt = 0
            for kk,vv in v.items():
                l = vv['l']
                r = vv['r']
                mlr[k].loc[kk] = [np.mean(l),np.mean(r)]
                slr[k].loc[kk] = [stats.sem(l),stats.sem(r)]
                cnt+=1


        for k in Conds: # keys : Cue, Desc, Cue_Desc
            for ii  in dat_meas:
                if ii=='Stats':
                    for jj in ['T','P','S']:
                        if unitNum == 0:
                            uni_LvR_Analyses[k][ii][jj] = pd.DataFrame(np.zeros((nUnits,3)),columns=LvR[k].index.values)
                        uni_LvR_Analyses[k]['Stats'][jj].loc[unitNum] = LvR[k][jj]
                else:
                    for jj in ['L','R']:
                        if unitNum == 0:
                            uni_LvR_Analyses[k][ii][jj] = pd.DataFrame(np.zeros((nUnits,3)),columns=LvR[k].index.values)
                        uni_LvR_Analyses[k]['Mean'][jj].loc[unitNum] = mlr[k][jj]
                        uni_LvR_Analyses[k]['SD'][jj].loc[unitNum] = slr[k][jj]

        all_dat_spl[unitNum] = dat_splits

    # reorg LvR to a pandas data frame with all the units
    CueDescFR_Dat = pd.DataFrame()
    for k in Conds:
        cnt = 0
        for kk in ['Mean','SD']:
            for kkk in ['L','R']:

                if kk=='Mean':
                    valName = 'MzFR_'+ kkk
                elif kk == 'SD':
                    valName = 'SzFR_' + kkk

                if cnt==0:
                    y = uni_LvR_Analyses[k][kk][kkk].copy()
                    y = y.reset_index()
                    y = y.melt(value_vars = uni_LvR_Analyses[k][kk][kkk].columns,id_vars='index',var_name='Seg',value_name= valName)
                    y['Cond'] = k
                else:
                    z = uni_LvR_Analyses[k][kk][kkk].copy()
                    z = z.reset_index()
                    z = z.melt(value_vars = uni_LvR_Analyses[k][kk][kkk].columns,id_vars='index',value_name= valName)
                    y[valName] = z[valName].copy()
                cnt+=1

        for jj in ['T','P','S']:
            z = uni_LvR_Analyses[k]['Stats'][jj].copy()
            z = z.reset_index()
            z = z.melt(value_vars = uni_LvR_Analyses[k]['Stats'][jj].columns ,id_vars='index', var_name = 'Seg', value_name = jj)
            y[jj] = z[jj]

        CueDescFR_Dat = pd.concat((CueDescFR_Dat,y))

    CueDescFR_Dat['Sig'] = CueDescFR_Dat['P']<0.05
    CueDescFR_Dat.rename(columns={'index':'unit'},inplace=True)

    return CueDescFR_Dat, all_dat_spl

def CueDesc_SegDecAnalysis(dat):
    nPe = 100
    nRepeats = 10
    nSh = 50
    njobs = 20

    trConds = dat['TrialConds']
    trDat = dat['TrialLongMat']
    nUnits = dat['TrialModelFits'].shape[0]

    gTrialsIDs = trConds['Good']
    Trials = trConds[gTrialsIDs].index.values
    nTrials = len(Trials)

    allZoneFR,unitIDs = reformatFRDat(dat,Trials)

    CoTrials =  trConds[gTrialsIDs & (trConds['Co']=='Co')].index.values
    InCoTrials = trConds[gTrialsIDs & (trConds['Co']=='InCo')].index.values

    nInCo = len(InCoTrials)
    TrSets = {}
    TrSets['all'] = np.arange(nTrials)
    _,idx,_=np.intersect1d(np.array(Trials),np.array(CoTrials),return_indices=True)
    TrSets['co'] = idx
    _,idx,_=np.intersect1d(np.array(Trials),np.array(InCoTrials),return_indices=True)
    TrSets['inco'] = idx

    cueVec = trConds.loc[gTrialsIDs]['Cues'].values
    descVec = trConds.loc[gTrialsIDs]['Desc'].values
    predVec = {'Cue':cueVec, 'Desc':descVec}

    nFeatures = {'h':np.arange(1),'a':np.arange(2),'center':np.arange(3),'be':np.arange(4),'int':np.arange(5),'cdfg':np.arange(6),'goal':np.arange(7)}

    def correctTrials_Decoder(train,test):
        res = pd.DataFrame(np.zeros((3,4)),columns=['Test','BAc','P','Z'])

        temp = mod.fit(X_train[train],y_train[train])

        res.loc[0,'Test'] = 'Model'
        y_hat = temp.predict(X_train[test])
        res.loc[0,'BAc'] = bac(y_train[test],y_hat)*100

        # shuffle for held out train set
        mod_sh = np.zeros(nSh)
        for sh in np.arange(nSh):
            y_perm_hat = np.random.permutation(y_hat)
            mod_sh[sh] = bac(y_train[test],y_perm_hat)*100
        res.loc[0,'Z'] = getPerm_Z(mod_sh, res.loc[0,'BAc'] )
        res.loc[0,'P'] = getPerm_Pval(mod_sh, res.loc[0,'BAc'] )

        # predictions on x test
        y_hat = temp.predict(X_test)
        res.loc[1,'Test'] = 'Cue'
        res.loc[1,'BAc'] = bac(y_test_cue,y_hat)*100

        res.loc[2,'Test'] = 'Desc'
        res.loc[2,'BAc'] = bac(y_test_desc,y_hat)*100

        # shuffles for ytest cue/desc
        cue_sh = np.zeros(nSh)
        desc_sh = np.zeros(nSh)
        for sh in np.arange(nSh):
            y_perm_hat = np.random.permutation(y_hat)
            cue_sh[sh] = bac(y_test_cue,y_perm_hat)*100
            desc_sh[sh] = bac(y_test_desc,y_perm_hat)*100

        res.loc[1,'Z'] = getPerm_Z(cue_sh, res.loc[1,'BAc'] )
        res.loc[1,'P'] = getPerm_Pval(cue_sh, res.loc[1,'BAc'] )

        res.loc[2,'Z'] = getPerm_Z(desc_sh, res.loc[2,'BAc'] )
        res.loc[2,'P'] = getPerm_Pval(desc_sh, res.loc[2,'BAc'] )

        return res

    def balancedCoIncoTrial_Decoder(pe,feats):

        res = pd.DataFrame(np.zeros((2,4)),columns=['Test','BAc','P','Z'])

        # sample correct trials to match the number of incorrect trials.
        samp_co_trials = np.random.choice(TrSets['co'],nInCo,replace=False)

        train = np.concatenate( (TrSets['inco'], samp_co_trials ))
        test = np.setdiff1d(TrSets['co'], samp_co_trials)

        X_train = allZoneFR.loc[train,feats].values
        X_test = allZoneFR.loc[test,feats].values

        Y_cue_train = predVec['Cue'][train]
        Y_desc_train = predVec['Desc'][train]

        Y_test = predVec['Cue'][test] # cue and desc trials are the on the test set.

        # model trained on the cue
        res.loc[0,'Test'] = 'Cue'
        cue_mod = mod.fit(X_train,Y_cue_train)
        y_cue_hat = cue_mod.predict(X_test)
        res.loc[0,'BAc']  = bac(Y_test,y_cue_hat)*100

        cue_sh = np.zeros(nSh)
        for sh in np.arange(nSh):
            y_perm = np.random.permutation(Y_test)
            cue_sh[sh] = bac(y_perm,y_cue_hat)*100

        res.loc[0,'Z'] = getPerm_Z(cue_sh, res.loc[0,'BAc'] )
        res.loc[0,'P'] = getPerm_Pval(cue_sh, res.loc[0,'BAc'] )

        # model trained on the desc
        res.loc[1,'Test'] = 'Desc'
        desc_mod = mod.fit(X_train,Y_desc_train)
        y_desc_hat = desc_mod.predict(X_test)
        res.loc[1,'BAc']  = bac(Y_test,y_desc_hat)*100

        desc_sh = np.zeros(nSh)
        for sh in np.arange(nSh):
            y_perm = np.random.permutation(Y_test)
            desc_sh[sh] = bac(y_perm,y_desc_hat)*100
        res.loc[1,'Z'] = getPerm_Z(cue_sh, res.loc[1,'BAc'] )
        res.loc[1,'P'] = getPerm_Pval(cue_sh, res.loc[1,'BAc'] )

        return res

    def IncoTrial_Decoder(train,test):

        res = pd.DataFrame(np.zeros((3,4)),columns=['Test','BAc','P','Z'])
        temp = mod.fit(X_train[train],y_train[train])

        res.loc[0,'Test'] = 'Model'
        y_hat = temp.predict(X_train[test])
        res.loc[0,'BAc'] = bac(y_train[test],y_hat)*100

        # shuffle for held out train set
        mod_sh = np.zeros(nSh)
        for sh in np.arange(nSh):
            y_perm_hat = np.random.permutation(y_hat)
            mod_sh[sh] = bac(y_train[test],y_perm_hat)*100
        res.loc[0,'Z'] = getPerm_Z(mod_sh, res.loc[0,'BAc'] )
        res.loc[0,'P'] = getPerm_Pval(mod_sh, res.loc[0,'BAc'] )

        # predictions on x test
        y_hat = temp.predict(X_test)
        res.loc[1,'Test'] = 'Cue'
        res.loc[1,'BAc'] = bac(y_test_cue,y_hat)*100

        res.loc[2,'Test'] = 'Desc'
        res.loc[2,'BAc'] = 100-res.loc[1,'BAc']

        # shuffles for ytest cue/desc
        cue_sh = np.zeros(nSh)
        for sh in np.arange(nSh):
            y_perm_hat = np.random.permutation(y_hat)
            cue_sh[sh] = bac(y_test_cue,y_perm_hat)*100

        res.loc[1,'Z'] = getPerm_Z(cue_sh, res.loc[1,'BAc'] )
        res.loc[1,'P'] = getPerm_Pval(cue_sh, res.loc[1,'BAc'] )

        res.loc[2,'Z'] = getPerm_Z(100-cue_sh, res.loc[2,'BAc'] )
        res.loc[2,'P'] = getPerm_Pval(100-cue_sh, res.loc[2,'BAc'] )

        return res

    with Parallel(n_jobs=njobs) as parallel:
        # correct trials Model:
        coModsDec = pd.DataFrame()
        popCoModsDec = pd.DataFrame()

        try:
            nFolds = 10
            y_train = predVec['Cue'][TrSets['co']]
            y_test_cue = predVec['Cue'][TrSets['inco']]
            y_test_desc = predVec['Desc'][TrSets['inco']]
            rskf = RepeatedStratifiedKFold(n_splits=nFolds,n_repeats=nRepeats, random_state=0)

            t0=time.time()
            for unitNum in np.arange(nUnits):
                for p,nF in nFeatures.items():

                    feats = unitIDs[unitNum][nF]
                    mod = lm.LogisticRegression(class_weight='balanced',C=1/np.sqrt(len(feats)))

                    X_train = allZoneFR.loc[TrSets['co'], feats ].values
                    X_test = allZoneFR.loc[TrSets['inco'], feats ].values

                    cnt=0

                    r = parallel(delayed(correctTrials_Decoder)(train,test) for train,test in rskf.split(X_train,y_train))
                    t1=time.time()

                    res = pd.DataFrame()
                    for jj in r:
                        res = pd.concat((jj,res))
                    res['Loc'] = p
                    res['-log(P)'] = -np.log(res['P'])
                    res['unit'] = unitNum

                    coModsDec = pd.concat((coModsDec,res))
                    print(end='.')
            coModsDec['Decoder'] = 'Correct'
            # -population
            for p,nF in nFeatures.items():
                feats=np.array([])
                for f in nF:
                    feats=np.concatenate((feats,np.arange(f,nUnits*7,7)))
                feats=feats.astype(int)
                mod = lm.LogisticRegression(class_weight='balanced',C=1/np.sqrt(len(feats)))

                X_train = allZoneFR.loc[TrSets['co'], feats ].values
                X_test = allZoneFR.loc[TrSets['inco'], feats ].values

                cnt=0
                r = parallel(delayed(correctTrials_Decoder)(train,test) for train,test in rskf.split(X_train,y_train))

                res = pd.DataFrame()
                for jj in r:
                    res = pd.concat((jj,res))
                res['Loc'] = p
                res['-log(P)'] = -np.log(res['P'])

                popCoModsDec = pd.concat((popCoModsDec,res))
                print(end='.')
            print('\nDecoding Correct Model Completed. Time  = {0:.2f}s \n'.format(time.time()-t0))
            popCoModsDec['Decoder'] = 'Correct'
        except:
            print('CorrectTrials Model Failed.')
            print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)

        # balanced correct/inco model:
        baModsDec = pd.DataFrame()
        popBaModsDec = pd.DataFrame()
        try:
            t0=time.time()
            for unitNum in np.arange(nUnits):
                for p,nF in nFeatures.items():
                    feats = unitIDs[unitNum][nF]
                    mod = lm.LogisticRegression(class_weight='balanced',C=1/np.sqrt(len(feats)))
                    r = parallel(delayed(balancedCoIncoTrial_Decoder)(pe, feats) for pe in np.arange(nPe))
                    res = pd.DataFrame()
                    for jj in r:
                        res = pd.concat((jj,res))
                    res['Loc'] = p
                    res['-log(P)'] = -np.log(res['P'])
                    res['unit'] = unitNum

                    baModsDec = pd.concat((baModsDec,res))
                    print(end='.')
            baModsDec['Decoder'] = 'Balanced'
            # -population
            for p,nF in nFeatures.items():
                feats=np.array([])
                for f in nF:
                    feats=np.concatenate((feats,np.arange(f,nUnits*7,7)))
                feats=feats.astype(int)
                mod = lm.LogisticRegression(class_weight='balanced',C=1/np.sqrt(len(feats)))
                r = parallel(delayed(balancedCoIncoTrial_Decoder)(pe, feats) for pe in np.arange(nPe))
                res = pd.DataFrame()
                for jj in r:
                    res = pd.concat((jj,res))
                res['Loc'] = p
                res['-log(P)'] = -np.log(res['P'])

                popBaModsDec = pd.concat((popBaModsDec,res))
                print(end='.')
            print('\nDecoding Balanced  Model Completed. Time  = {0:.2f}s \n'.format(time.time()-t0))
            popBaModsDec['Decoder'] = 'Balanced'
        except:
            print('Balanced Model Failed.')
            print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)

        # incorrect trials model:
        InCoModsDec = pd.DataFrame()
        popInCoModsDec = pd.DataFrame()
        try:
            t0=time.time()
            nFolds = 5
            y_train = predVec['Cue'][TrSets['inco']]
            y_test_cue = predVec['Cue'][TrSets['co']]
            y_test_desc = predVec['Desc'][TrSets['co']]
            rskf = RepeatedStratifiedKFold(n_splits=nFolds,n_repeats=nRepeats, random_state=0)

            for unitNum in np.arange(nUnits):
                for p,nF in nFeatures.items():
                    feats = unitIDs[unitNum][nF]
                    mod = lm.LogisticRegression(class_weight='balanced',C=1/np.sqrt(len(feats)))

                    X_train = allZoneFR.loc[TrSets['inco'], feats ].values
                    X_test = allZoneFR.loc[TrSets['co'], feats ].values

                    cnt=0
                    r = parallel(delayed(IncoTrial_Decoder)(train,test) for train,test in rskf.split(X_train,y_train))
                    res = pd.DataFrame()
                    for jj in r:
                        res = pd.concat((jj,res))
                    res['Loc'] = p
                    res['-log(P)'] = -np.log(res['P'])
                    res['unit'] = unitNum

                    InCoModsDec = pd.concat((InCoModsDec,res))
                    print(end='.')
            InCoModsDec['Decoder'] = 'Incorrect'

            #-population
            for p,nF in nFeatures.items():
                feats=np.array([])
                for f in nF:
                    feats=np.concatenate((feats,np.arange(f,nUnits*7,7)))
                feats=feats.astype(int)
                mod = lm.LogisticRegression(class_weight='balanced',C=1/np.sqrt(len(feats)))

                X_train = allZoneFR.loc[TrSets['inco'], feats ].values
                X_test = allZoneFR.loc[TrSets['co'], feats ].values

                cnt=0
                r = parallel(delayed(IncoTrial_Decoder)(train,test) for train,test in rskf.split(X_train,y_train))
                res = pd.DataFrame()
                for jj in r:
                    res = pd.concat((jj,res))
                res['Loc'] = p
                res['-log(P)'] = -np.log(res['P'])

                popInCoModsDec = pd.concat((popInCoModsDec,res))
                print(end='.')
            print('\nDecoding Incorrect Model Completed. Time  = {0:.2f}s \n'.format(time.time()-t0))

            popInCoModsDec['Decoder'] = 'Incorrect'
        except:
            print('Incorrect Model Failed.')
            print ("Error", sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)

        # group results.
        singCellDec = pd.concat((coModsDec,baModsDec,InCoModsDec))
        popDec = pd.concat((popCoModsDec,popBaModsDec,popInCoModsDec))

        singCellDecSummary = singCellDec.groupby(['Loc','Test','unit','Decoder']).mean()
        singCellDecSummary = singCellDecSummary.reset_index()
        singCellDecSummary['Test'] = pd.Categorical(singCellDecSummary['Test'],categories=['Model','Cue','Desc'],ordered=True)
        singCellDecSummary.sort_values('Test',inplace=True)
        singCellDecSummary['Loc'] = pd.Categorical(singCellDecSummary['Loc'],categories=nFeatures.keys(),ordered=True)
        singCellDecSummary.sort_values('Loc',inplace=True)

    return singCellDec,singCellDecSummary, popDec

def reformatFRDat(dat,Trials):
    trDat = dat['TrialLongMat']
    nCells = len(dat['ids']['cells'])
    nMua = len(dat['ids']['muas'])
    nUnits = nCells+nMua
    nTrials = len(Trials)

    allZoneFR = np.zeros((nTrials,7*nUnits))
    unitIDs = np.zeros((nUnits,7),dtype=int)

    for t in np.arange(nTrials):
        subset = (trDat['trID']==Trials[t]) & (trDat['IO']=='Out')
        cnt =0
        for c in np.arange(nCells):
            cc = c*7
            unitIDs[cnt] = np.arange(cc,cc+7,dtype=int)
            allZoneFR[t,(cc):(cc+7)] = dat['TrialFRLongMat']['cell_'+str(c)][subset].values[:7]
            cnt+=1
        cc = nCells*7
        for m in np.arange(nMua):
            mm = m*7+cc
            unitIDs[cnt] = np.arange(mm,mm+7,dtype=int)
            allZoneFR[t,(mm):(mm+7)] = dat['TrialFRLongMat']['mua_'+str(m)][subset].values[:7]
            cnt+=1
    allZoneFR = pd.DataFrame(allZoneFR)
    return allZoneFR, unitIDs

def datBarPlots(m,s,ax,xlabels,colors):
    nSubTypes,nGroups = m.shape
    w = 0.4
    alpha = 0.5
    x1 = np.arange(nSubTypes)-w/2
    ax.bar(x1,m[:,0],w,label='L',color=colors[0],alpha=alpha)
    x2 = np.arange(nSubTypes)+w/2
    ax.bar(x2,m[:,1],w,label='R',color=colors[1],alpha=alpha)

    for ss in np.arange(nSubTypes):
        ax.plot([x1[ss],x1[ss]], [-s[ss,0],s[ss,0]]+m[ss,0],'kd-',lw=3,markersize=3)
        ax.plot([x2[ss],x2[ss]], [-s[ss,1],s[ss,1]]+m[ss,1],'kd-',lw=3,markersize=3)

    ax.set_xticks(np.arange(nSubTypes))
    ax.set_xticklabels(xlabels)
    return ax

def datScatPlot(dat,ax,colors):
    w = 0.4
    alpha=0.5
    mSize = 3
    nSubTypes=len(dat.keys())
    x1 = np.arange(nSubTypes)-w/2
    x2 = np.arange(nSubTypes)+w/2

    cnt=0
    for k,v in dat.items():
        nDat = len(v['l'])
        xx1 = x1[cnt]+(np.random.rand(nDat)-0.5)*w/2
        nDat = len(v['r'])
        xx2 = x2[cnt]+(np.random.rand(nDat)-0.5)*w/2
        ax.scatter(xx1,v['l'],marker='o',color=colors[0],alpha=alpha,s=mSize)
        ax.scatter(xx2,v['r'],marker='o',color=colors[1],alpha=alpha,s=mSize)
        cnt+=1
    return ax

def plotCueVDes(CueDescFR_Dat,sePaths):

    f,ax = plt.subplots(4,3,figsize=(17,23))
    cnt1 = 0
    for c in ['Cue','Desc','Ts']:
        cnt2 = 0
        for s in Segs:
            if c == 'Ts':
                subset = (CueDescFR_Dat['Cond']=='Cue') & (CueDescFR_Dat['Seg']==s)
                x = CueDescFR_Dat[subset]['T']
                subset = (CueDescFR_Dat['Cond']=='Desc') & (CueDescFR_Dat['Seg']==s)
                y= CueDescFR_Dat[subset]['T']
            else:
                subset = (CueDescFR_Dat['Cond']==c) & (CueDescFR_Dat['Seg']==s)
                x = CueDescFR_Dat[subset]['MzFR_R']
                y = CueDescFR_Dat[subset]['MzFR_L']

            plt.sca(ax[cnt1,cnt2])
            plt.scatter(x=x,y=y)
            #ax[cnt1,cnt2]=sns.scatterplot(x=x,y=y,ax=ax[cnt1,cnt2])
            xlim = ax[cnt1,cnt2].get_xlim()
            ylim =ax[cnt1,cnt2].get_ylim()
            mi = np.min([xlim,ylim])
            ma = np.max([xlim,ylim])
            ax[cnt1,cnt2].plot([mi,ma], [mi, ma], ls="--", c=".3")
            ax[cnt1,cnt2].set_xlim(xlim)
            ax[cnt1,cnt2].set_ylim(ylim)

            _,p=stats.ttest_rel(x,y)
            if p<0.05:
                pp = '*'
            else:
                pp = ''

            if (cnt1==1) or (cnt1==0):
                ax[cnt1,cnt2].set_xlabel('R [zFT]' +pp)

            if cnt2==0:
                ax[cnt1,cnt2].set_ylabel(c + ' L [zFR]')
            else:
                ax[cnt1,cnt2].set_ylabel('')

            if cnt1==2:
                ax[cnt1,cnt2].set_xlabel('Cue LvR [t]' + pp)
                if cnt2==0:
                    ax[cnt1,cnt2].set_ylabel('Des LvR [t]')

            if cnt1==0:
                ax[0,cnt2].set_title('Segment ' + s)

            cnt2+=1
        cnt1+=1

    c = 'Cue_Desc'
    cnt2 = 0
    for s in ['Co_Arm','L_Arm','R_Arm']:
        subset = (CueDescFR_Dat['Cond']==c) & (CueDescFR_Dat['Seg']==s)
        x = CueDescFR_Dat[subset]['MzFR_R']
        subset = (CueDescFR_Dat['Cond']==c) & (CueDescFR_Dat['Seg']==s)
        y = CueDescFR_Dat[subset]['MzFR_L']
        z = np.abs(CueDescFR_Dat[subset]['T'])

        plt.sca(ax[cnt1,cnt2])
        plt.scatter(x=x,y=y,s=z*10)
        #ax[cnt1,cnt2]=sns.scatterplot(x=x,y=y,ax=ax[cnt1,cnt2])
        xlim = ax[cnt1,cnt2].get_xlim()
        ylim =ax[cnt1,cnt2].get_ylim()
        mi = np.min([xlim,ylim])
        ma = np.max([xlim,ylim])
        ax[cnt1,cnt2].plot([mi,ma], [mi, ma], ls="--", c=".3")
        ax[cnt1,cnt2].set_xlim(xlim)
        ax[cnt1,cnt2].set_ylim(ylim)

        _,p=stats.ttest_rel(x,y)
        if p<0.05:
            pp = '*'
        else:
            pp = ''

        ax[cnt1,cnt2].set_xlabel('R [zFR]' +pp)
        if cnt2==0:
            ax[cnt1,cnt2].set_ylabel('L [zFR]')

        ax[cnt1,cnt2].set_title('Segment ' + s)

        plt.close(f)

    fn = sePaths['CueDescPlots'] / 'SeUnits_CueVDec.jpeg'
    f.savefig(str(fn),dpi=150, bbox_inches='tight',pad_inches=0.2)
    plt.close(f)

def plotUnitRvL(CueDescFR_Dat,all_dat_spl,sePaths):
    nUnits = len(all_dat_spl)

    for unitNum in np.arange(nUnits):
        f,ax=plt.subplots(3,1,figsize=(6,9))
        #bars
        pal = sns.xkcd_palette(['kelly green','light purple'])
        cnt = 0
        for a in ['Cue','Desc','Cue_Desc']:
            subset = (CueDescFR_Dat['Cond']==a) & (CueDescFR_Dat['unitNum']==unitNum)
            xlabs = CueDescFR_Dat[subset]['Seg'].values + CueDescFR_Dat[subset]['S'].T.values
            ax[cnt] = datBarPlots(CueDescFR_Dat[subset][['MzFR_L','MzFR_R']].values , CueDescFR_Dat[subset][['SzFR_L','SzFR_R']].values ,ax[cnt],xlabs,pal)
            cnt+=1
        #scatter
        pal = np.array(sns.xkcd_palette(['emerald green','medium purple']))
        cnt = 0
        for a in ['Cue','Desc','Cue_Desc']:
            ax[cnt] =datScatPlot(all_dat_spl[unitNum][a],ax[cnt],pal)
            ax[cnt].set_ylabel(a + ' [zFR]')
            cnt+=1

        fn = sePaths['CueDescPlots'] / ('UnitID-{}_RvL.jpeg'.format(unitNum) )
        f.savefig(str(fn),dpi=150, bbox_inches='tight',pad_inches=0.2)
        plt.close(f)

def plotMultipleDecoderResults(data,plotAll=False,ci='sd'):
    f,ax = plt.subplots(2,3,figsize=(18,10))

    cols = ['Correct','Balanced','Incorrect']

    jj=0
    for c in cols:
        subset = (data['Decoder'] == c)

        if c=='Balanced':
            pal =sns.color_palette(desat=.9)[1:3]
            hue_order = ['Cue','Desc']
        else:
            pal =sns.color_palette(desat=.9)[:3]
            hue_order = ['Model','Cue','Desc']

        ii = 0
        ax[ii,jj].set_ylim([-5,110])
        ax[ii,jj].set_yticks([0,25,50,75,100])
        ax[ii,jj].set_xlim([-0.2,6.2])
        ax[ii,jj].axhline(y=50,linestyle='--',color=0.3*np.ones(3),alpha=0.5)
        ax[ii,jj] = sns.pointplot(x='Loc',y='BAc',hue='Test',dodge = 0.3,sort=False,ci=ci,palette = pal, data= data[ subset], ax=ax[ii,jj], hue_order = hue_order)
        if plotAll:
            ax[ii,jj] = sns.stripplot(x='Loc',y='BAc',hue='Test', dodge = True ,alpha=0.3,palette  = pal, data= data[subset], ax=ax[ii,jj], hue_order = hue_order)

        ax[ii,jj].set_title(c)
        ax[ii,jj].set_xlabel('')
        l=ax[ii,jj].get_legend()
        l.set_visible(False)

        ii=1
        ax[ii,jj] = sns.pointplot(x='Loc',y='Z',hue='Test',dodge = 0.3,sort=False,ci=ci,palette  = pal, data= data[ subset ],ax=ax[ii,jj], hue_order = hue_order)
        if plotAll:
            ax[ii,jj] = sns.stripplot(x='Loc',y='Z',hue='Test', dodge = True ,alpha=0.3,palette  = pal,data= data[subset],ax=ax[ii,jj], hue_order = hue_order)

        l=ax[ii,jj].get_legend()
        l.set_visible(False)
        if jj==2:
            handles, labels = ax[ii,jj].get_legend_handles_labels()
            if c =='balanced':
                plt.legend(handles[:2],labels[:2],bbox_to_anchor=(1.02, 1), loc=3, borderaxespad=0.,frameon=False,title='Test')
            else:
                plt.legend(handles[:3],labels[:3],bbox_to_anchor=(1.02, 1), loc=3, borderaxespad=0.,frameon=False,title='Test')

        ax[ii,jj].axhline(y=0,linestyle='--',color=0.3*np.ones(3),alpha=0.5)

        if jj==0:
            ax[0,jj].set_ylabel(' BAc ')
            ax[1,jj].set_ylabel(' Z [BAc] ')
        else:
            ax[0,jj].set_ylabel('')
            ax[1,jj].set_ylabel('')

        jj+=1
    return f, ax

def getSigLevel(pvals):
    s = np.array(['']*len(pvals))
    cnt = 0
    for p in pvals:
        if p<0.001:
            s[cnt] = '***'
        elif p<0.01:
            s[cnt] = '**'
        elif p<0.05:
            s[cnt] = '*'
        else:
            s[cnt] = ''
        cnt+=1
    return s

def getPerm_Pval(nullDist,score):
    nP = len(nullDist)
    p = 1-np.sum(score>nullDist)/nP
    if p==0:
        p = 1/(nP+1)
    if p==1:
        p = 1-1/(nP+1)
    return p

def getPerm_Z(nullDist,score):
    tol = 0.001
    m = np.nanmean(nullDist)
    s = np.nanstd(nullDist + np.random.randn(len(nullDist))*tol)

    return (score-m)/s
