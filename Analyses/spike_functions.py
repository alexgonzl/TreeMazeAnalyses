import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats
from scipy.interpolate import CubicSpline

from pathlib import Path
import os,sys
import h5py

sys.path.append('../PreProcessing/')
sys.path.append('../TrackingAnalyses/')
sys.path.append('../Lib/')
from filters_ag import *

import pre_process_neuralynx as PPN
import TreeMazeFunctions as TMF


def get_spikes(IDs,cluster_path):
    cluster_path = Path(cluster_path)
    nUnits=0
    for tt_id,cl_id in IDs.items():
        nUnits+=len(cl_id)
        pass
    spikes = {}
    spikes['nUnits'] = nUnits
    for tt_id,cl_ids in IDs.items():
        if len(cl_ids)>0:
            ttPath = Path(cluster_path,'tt_'+str(tt))
            sp_times = np.load(str(ttPath/'spike_times.npy'))
            clusters = np.load(str(ttPath/'spike_clusters.npy'))
            spikes[tt_id]={}
            for cl in cl_ids:
                spikes[tt_id][cl]=sp_times[clusters==cl]
    return spikes

def bin_spikes(spikes,orig_time,time_vector):
    step = time_vector[1]-time_vector[0]
    nTimePoints = len(time_vector)
    sp_bins = np.zeros((spikes['nUnits'],nTimePoints))
    sp_ids = {}
    cnt = 0
    for tt, cl_ids in spikes:
        for cl in cl_ids:
            sp_ids[cnt] = (tt,cl)
            sp_bins[cnt] = np.histogram(orig_time[spikes[tt][cl]],bins=nTimePoints)

    return sp_bins,sp_ids
