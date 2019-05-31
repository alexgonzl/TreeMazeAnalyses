import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats
import sys, os, time
from pathlib import Path
import h5py

sys.path.append('../PreProcessing/')
sys.path.append('../Lib/')
sys.path.append('../Analyses/')

from pre_process_neuralynx import *
from filters_ag import *
import nept

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from shapely.geometry import Point
from shapely.geometry.polygon import LinearRing, Polygon
from collections import Counter
from descartes import PolygonPatch

import pre_process_neuralynx as PPN
import TreeMazeFunctions as TMF
import spike_functions as SF
import spatial_tuning as ST
import stats_functions as StatsF

################################################################################
# Plot Functions
################################################################################
def plotPoly(poly,ax,alpha=0.3,color='g'):
    p1x,p1y = poly.exterior.xy
    ax.plot(p1x, p1y, color=[0.5, 0.5,0.5],
        linewidth=2,)
    ring_patch = PolygonPatch(poly, fc=color, ec='none', alpha=alpha)
    ax.add_patch(ring_patch)

def plotCounts(counts, names,ax):
    nX = len(names)
    ab=sns.barplot(x=np.arange(nX),y=counts,ax=ax, ci=[],facecolor=(0.4, 0.6, 0.7, 1))
    ax.set_xticks(np.arange(nX))
    ax.set_xticklabels(names)
    sns.set_style("whitegrid")
    sns.despine(left=True)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    return ax

def plotBehWindow(time,dat,names,ax):
    sns.heatmap(dat,ax=ax,yticklabels=names,cbar=0,cmap='Greys_r',vmax=1.1)
    ax.hlines(np.arange(len(names)+1), *ax.get_xlim(),color=(0.7,0.7,0.7,1))
    x=ax.get_xticks().astype(int)
    x=np.linspace(x[0],x[-1], 6, endpoint=False).astype(int)
    x=x[1::]
    ax.set_xticks(x)
    ax.vlines(x,*ax.get_ylim(),color=(0.3,0.3,0.3,1),linestyle='-.')
    _=ax.set_xticklabels(np.round(time[x]).astype(int))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    return ax

def plotBehavZonesWindowAndSpikes(time,behav,bin_spikes):
    f,(a1,a2)=plt.subplots(2,1, figsize=(12,12))
    a1.set_position([0.125, 0.4, 0.775, 0.4])
    a1=plotBehWindow(time,behav,ZonesNames,a1)
    a1.set_xticks([])

    nCells = bin_spikes.shape[0]
    a2=plotBehWindow(time,bin_spikes,np.arange(nCells).astype(str),a2)
    a2.set_xlabel('Time[s]')
    a2.set_ylabel('Cell Number')

    a2.set_position([0.125, 0.2, 0.775, 0.18])

    yt=np.linspace(0,nCells,5, endpoint=False).astype(int)
    a2.set_yticks(yt)
    a2.set_yticklabels(yt.astype(str))
    for tick in a2.get_yticklabels():
        tick.set_rotation(0)
    return f,a1,a2

def plotTM_Trace(ax,x,y,bin_spikes=[], plot_zones=1, plot_raw_traces=0):
    if plot_zones:
        for zo in TMF.MazeZonesGeom.keys():
            plotPoly(TMF.MazeZonesGeom[zo],ax,alpha=0.2)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
    if plot_raw_traces:
        ax.scatter(x,y,0.5,marker='D',color='grey',alpha=0.05)
    if len(bin_spikes)==len(x):
        ax.scatter(x,y,s=bin_spikes, alpha=0.02, color = 'r')
    ax.set_axis_off()
    ax.set_xlim(TMF.x_limit)
    ax.set_ylim(TMF.y_limit)

    return ax

def plotZonesHeatMap(ax,cax,data,zones=TMF.ZonesNames,cmap='div',alpha=1):
    if cmap=='div':
        cDat,colArray =  getDatColorMap(data)
        cMap = plt.get_cmap('RdBu_r')
    else:
        cDat,colArray =  getDatColorMap(data,col_palette='YlOrBr_r',div=False)
        cMap = plt.get_cmap('YlOrBr_r')
    cnt=0
    for zo in zones:
        plotPoly(TMF.MazeZonesGeom[zo],ax,color=cDat[cnt],alpha=alpha)
        cnt+=1
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    ax.set_axis_off()
    ax.set_xlim(TMF.x_limit)
    ax.set_ylim(TMF.y_limit)
    ax.axis('equal')

    cNorm = mpl.colors.Normalize(vmin=colArray[0],vmax=colArray[-1])
    sm = plt.cm.ScalarMappable(cmap=cMap,norm=cNorm)
    sm.set_array([])

    cbar = plt.colorbar(sm,cax=cax)
    cax.yaxis.set_tick_params(right=False)

    return ax,cax

def getDatColorMap(data, nBins = 25, col_palette="RdBu_r",div=True):

    if div:
        maxV = np.ceil(np.max(np.abs(data))*100)/100
        colArray = np.linspace(-maxV,maxV,nBins-1)
    else:
        maxV = np.ceil(np.max(data)*100)/100
        minV = np.ceil(np.min(data)*100)/100
        colArray = np.linspace(minV,maxV,nBins-1)
    x = np.digitize(data,colArray).astype(int)
    colMap = np.array(sns.color_palette(col_palette, nBins))
    return colMap[x],colArray

def plotHeatMap(ax,cax,img,cmap='viridis', colbar_label = 'FR [sp/s]', smooth=True,robust=False,w=4,s=1):
    if smooth:
        img = ST.getSmoothMap(img,w,s)
    with sns.plotting_context(font_scale=2):
        ax=sns.heatmap(img.T,xticklabels=[],yticklabels=[],ax=ax,square=False, robust=robust, cbar_ax=cax, cmap=cmap,cbar_kws={'label': colbar_label})
        ax.invert_yaxis()
    return ax

def plotMaze_XY(x,y):
    f,a1=plt.subplots(1,1, figsize=(10,10))
    sns.set_style("white")
    for zo in TMF.MazeZonesGeom.keys():
        plotPoly(TMF.MazeZonesGeom[zo],a1)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    #a1.plot(PosDat['x'],PosDat['y'],alpha=0.1,color='k',linewidth=0.1)
    a1.scatter(x,y,20, alpha=0.005,color='k')
    a1.set_xlabel('x-position [mm]')
    a1.set_ylabel('y-position [mm]')
    a1.axis('equal')
    #a1.grid()
    a1.hlines(-150,-800,800,color=(0.7,0.7,0.7),linewidth=2)
    a1.vlines(-850,0,1400,color=(0.7,0.7,0.7),linewidth=2)
    a1.set_xlim([-800,800])
    a1.set_xticks([-800,0,800])
    a1.set_yticks([0,700,1400])
    a1.set_ylim([-160,1500])
    return f

def plotMazeZoneCounts(PosMat):
    f,a1=plt.subplots(1,1, figsize=(12,6))
    with sns.axes_style("whitegrid"):
        counts = np.sum(PosMat)
        a1 = plotCounts(counts/1000,TMF.ZonesNames,a1)
        a1.set_xlabel('Animal Location')
        a1.set_ylabel('Sample Counts [x1000]')
    return f

def plotEventCounts(EventMat):
    f,a1=plt.subplots(1,1, figsize=(12,6))
    ev_subset = ['RH','RC','R1','R2','R3','R4','DH','DC','D1','D2','D3','D4','CL','CR']
    counts = np.sum(EventMat[ev_subset]/1000,0)
    with sns.axes_style("whitegrid"):
        a1 = plotCounts(counts,ev_subset,a1)
        a1.set_yscale('log')
        a1.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        a1.set_yticks([1,10])
        a1.set_yticklabels([1,10])
        a1.set_xlabel('Event Type')
        a1.set_ylabel('Sample Counts [x1000]')

    return f
