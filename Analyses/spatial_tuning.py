import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import seaborn as sns

from pathlib import Path
import os,sys
import h5py
import sys

sys.path.append('../PreProcessing/')
sys.path.append('../TrackingAnalyses/')
sys.path.append('../Lib/')
sys.path.append('../Analyses/')
import TreeMazeFunctions as TMF

font = {'family' : 'sans-serif',
        'size'   : 20}

plt.rc('font', **font)


def getSmoothMap(fr_map,w=4,s=1):
    fr_smoothed = ndimage.filters.median_filter(fr_map,w)
    fr_smoothed = smooth_hist2d(fr_smoothed,w,s)
    return fr_smoothed

def smooth_hist2d(img,w,sigma=1.5):
    trunc = (((w - 1)/2)-0.5)/sigma
    return ndimage.filters.gaussian_filter(img,sigma,mode='constant',truncate=trunc)

def getPositionMat(x,y,xlimits, ylimits, spacing=30):
    xedges=np.arange(xlimits[0],xlimits[1]+1,spacing)
    yedges=np.arange(ylimits[0],ylimits[1]+1,spacing)
    position_2d,_,_ = np.histogram2d(x,y, bins=[xedges, yedges])
    return position_2d, xedges, yedges

def binSpikesToXY(bin_spikes,x,y):
    maxBinSpikes = np.max(bin_spikes)
    xPosSpikes = []
    yPosSpikes = []
    for nsp in np.arange(1,maxBinSpikes+1):
        xPosSpikes+=x[bin_spikes==nsp].tolist()*int(nsp)
        yPosSpikes+=y[bin_spikes==nsp].tolist()*int(nsp)
    assert len(xPosSpikes)==np.sum(bin_spikes), 'Spikes To Position Mismatch'
    return xPosSpikes,yPosSpikes

def binSpikesToZone(bin_spikes,Zones):
    # returns a vector of where each spike occured.
    maxBinSpikes = np.max(bin_spikes)
    zoneSpikes = []
    for nsp in np.arange(1,maxBinSpikes+1):
        zoneSpikes+=Zones[bin_spikes==nsp].tolist()*int(nsp)
    return zoneSpikes

def getZoneSpikeMaps(sp,Zones):
    ZoneSp = binSpikesToZone(sp,Zones)
    ZoneSpCounts = np.bincount(ZoneSp)
    return ZoneSpCounts

def getPosBinSpikeMaps(spikes,PosDat,spacing=25):
    xSp,ySp = binSpikesToXY(spikes,PosDat['x'],PosDat['y'])
    spikes_pos,_,_=get_position_mat(xSp,ySp,TMF.x_limit,TMF.y_limit,spacing)
    return spikes_pos

def getPosBinFRMaps(spikes_pos,occ_time):
    occ_time2 = np.array(occ_time)
    occ_time2[occ_time==0]=np.nan
    fr_pos=spikes_pos/occ_time2
    fr_pos[np.isnan(fr_pos)]=0
    return fr_pos

def getValidMovingSamples(speed, sp_thr = [5,2000]):
    # subsample to get moving segments
    return np.logical_and(speed>=sp_thr[0],speed<=sp_thr[1])

def getDirZoneSpikeMaps(spikes, PosDat, sp_thr = [5,2000]):
    SegSeq = PosDat['SegDirSeq']

    # subsample to get moving segments
    valid_moving = getValidMovingSamples(PosDat['Speed'])
    valid_samps = np.logical_and(valid_moving,SegSeq>0)

    MovSegSeq=np.array(SegSeq)
    dir_seq = MovSegSeq[valid_samps]-1
    seqInfo = getSeqInfo(dir_seq,PosDat['step'])

    dir_spikes = spikes[valid_samps]
    spikesByZoneDir = getZoneSpikeMaps(dir_spikes,dir_seq)
    return spikesByZoneDir, seqInfo

def getSeqInfo(Seq,step=0.02):
    fields = ['counts','time','prob']
    seqInfo = pd.DataFrame(np.full((3,TMF.nZones),np.nan),columns=TMF.ZonesNames,index=fields)

    counts = np.bincount(Seq.astype(int),minlength=TMF.nZones)
    seqInfo.loc['counts']  = counts
    seqInfo.loc['time'] = counts*step
    seqInfo.loc['prob'] = counts/np.sum(counts)

    return seqInfo

def getTM_OccupationInfo(PosDat,spacing=25,occ_time_thr=0.1):
    occ_counts,xed,yed = getPositionMat(PosDat['x'],PosDat['y'],TMF.x_limit,TMF.y_limit,spacing)
    occ_time = occ_counts*PosDat['step']
    occ_mask = occ_time>=occ_time_thr

    OccInfo = {}
    OccInfo['time'] = occ_time*occ_mask
    OccInfo['counts'] = occ_counts*occ_mask
    OccInfo['prob'] = occ_counts/np.nansum(occ_counts)
    return OccInfo
