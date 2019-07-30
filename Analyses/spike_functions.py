import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats
from scipy.interpolate import CubicSpline

from pathlib import Path
import os,sys, json
import h5py

sys.path.append('../PreProcessing/')
sys.path.append('../TrackingAnalyses/')
sys.path.append('../Lib/')
from filters_ag import *

import pre_process_neuralynx as PPN
import TreeMazeFunctions as TMF

def getSessionSpikes(sessionPaths, overwrite=0):

    if (not sessionPaths['Cell_Spikes'].exists()) | overwrite:
        print('Spikes Files not Found or overwrite=1, creating them.')

        animal = sessionPaths['animal']
        task = sessionPaths['task']
        date = sessionPaths['date']

        with sessionPaths['ClusterTable'].open() as f:
            CT = json.load(f)
        sessionCellIDs = CT[animal][date][task]['cell_IDs']
        sessionMuaIDs = CT[animal][date][task]['mua_IDs']

        cell_spikes = get_TT_spikes(sessionCellIDs,sessionPaths['Clusters'])
        mua_spikes = get_TT_spikes(sessionMuaIDs,sessionPaths['Clusters'])

        with sessionPaths['Cell_Spikes'].open(mode='w') as f:
            json.dump(cell_spikes,f,indent=4)
        with sessionPaths['Mua_Spikes'].open(mode='w') as f:
            json.dump(mua_spikes,f,indent=4)
    else:
        with sessionPaths['Cell_Spikes'].open() as f:
            cell_spikes=json.load(f)
        with sessionPaths['Mua_Spikes'].open() as f:
            mua_spikes = json.load(f)
        print('Loaded Spike Files.')

    return cell_spikes, mua_spikes

def getSessionBinSpikes(sessionPaths,resamp_t, overwrite=0):
    if (not sessionPaths['Cell_Bin_Spikes'].exists()) | overwrite:
        print('Binned Spikes Files not Found or overwrite=1, creating them.')

        cell_spikes, mua_spikes = getSessionSpikes(sessionPaths, overwrite)

        cell_bin_spikes,cell_ids = bin_TT_spikes(cell_spikes,resamp_t,origSR=sessionPaths['SR'])
        mua_bin_spikes,mua_ids = bin_TT_spikes(mua_spikes,resamp_t,origSR=sessionPaths['SR'])

        ids = {}
        ids['cells'] = cell_ids
        ids['muas'] = mua_ids

        np.save(sessionPaths['Cell_Bin_Spikes'],cell_bin_spikes)
        np.save(sessionPaths['Mua_Bin_Spikes'],mua_bin_spikes)
        with sessionPaths['Spike_IDs'].open(mode='w') as f:
            json.dump(ids,f,indent=4)
        print('Bin Spike File Creation and Saving Completed.')
    else:
        print('Loading Spikes...')
        cell_bin_spikes=np.load(sessionPaths['Cell_Bin_Spikes'])
        mua_bin_spikes=np.load(sessionPaths['Mua_Bin_Spikes'])
        with sessionPaths['Spike_IDs'].open() as f:
            ids = json.load(f)
        print('Binned Spike Files Loaded.')

    return cell_bin_spikes, mua_bin_spikes, ids

def getSessionFR(sessionPaths,overwrite=0):
    if (not sessionPaths['Cell_FR'].exists()) | overwrite:
        print('Firing Rate Files Not Found or overwrite=1, creating them.')

        cell_bin_spikes, mua_bin_spikes, ids = getSessionBinSpikes(sessionPaths, overwrite)

        nCells,nTimePoints = cell_bin_spikes.shape
        cell_FR = np.zeros((nCells,nTimePoints))
        for cell in np.arange(nCells):
            cell_FR[cell] = smoothSpikesTrain(cell_bin_spikes[cell])

        nΜUAs,nTimePoints = mua_bin_spikes.shape
        mua_FR = np.zeros((nΜUAs,nTimePoints))
        for cell in np.arange(nΜUAs):
            mua_FR[cell] = smoothSpikesTrain(mua_bin_spikes[cell])

        np.save(sessionPaths['Cell_FR'],cell_FR)
        np.save(sessionPaths['Mua_FR'],mua_FR)

        print('Spike File Creation and Saving Completed.')
    else:
        print('Loading FRs ...')
        cell_FR=np.load(sessionPaths['Cell_FR'])
        mua_FR=np.load(sessionPaths['Mua_FR'])

        print('FR Loaded.')
    return cell_FR, mua_FR

def get_TT_spikes(IDs,cluster_path):
    cluster_path = Path(cluster_path)
    nUnits=0
    for tt_id,cl_id in IDs.items():
        nUnits+=len(cl_id)
        pass
    spikes = {}
    spikes['nUnits'] = nUnits
    for tt_id,cl_ids in IDs.items():

        if len(cl_ids)>0:
            ttPath = Path(cluster_path,'tt_'+str(tt_id))
            sp_times = np.load(str(ttPath/'spike_times.npy'))
            clusters = np.load(str(ttPath/'spike_clusters.npy'))
            spikes[str(tt_id)]={}
            for cl in cl_ids:
                spikes[str(tt_id)][str(cl)]=sp_times[clusters==cl].tolist()
    return spikes

def bin_TT_spikes(spikes,resamp_t,origSR=32000):
    orig_time = np.arange(resamp_t[0],resamp_t[-1],1/origSR)
    step = resamp_t[1]-resamp_t[0]
    nOrigTimePoints = len(orig_time)
    nTimePoints = len(resamp_t)
    sp_bins = np.zeros((spikes['nUnits'],nTimePoints))
    sp_ids = {}
    cnt = 0
    for tt,cl_ids in spikes.items():
        if tt!='nUnits':
            for cl in cl_ids:
                try:
                    sp = np.array(spikes[tt][cl])
                    out_of_record_spikes = sp>=nOrigTimePoints
                    if np.any(out_of_record_spikes):
                        sp = np.delete(sp,np.where(out_of_record_spikes)[0])
                    sp_ids[cnt] = (tt,cl)
                    #print(type(sp[0]))
                    sp_bins[cnt],_ = np.histogram(orig_time[sp],bins=nTimePoints)
                except:
                    print("Error processing Tetrode {}, Cluster {}".format(tt,cl))
                    pass
                cnt+=1

    return sp_bins,sp_ids

def smoothSpikesTrain(bin_spikes,step=0.02):
    lfwin = np.round(1.0/step).astype(int)
    return signal.filtfilt(np.ones(lfwin)/lfwin,1,bin_spikes/step)

def getSpikeList(spikes,ids,cell_num):
    t,c = ids[str(cell_num)]
    return spikes[str(t)][str(c)]
