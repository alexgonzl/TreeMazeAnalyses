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
from seaborn.utils import remove_na
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.text import Text
from matplotlib import transforms, lines
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties

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
        linewidth=1.5)
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
    a1=plotBehWindow(time,behav,TMF.ZonesNames,a1)
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
        ax.scatter(x,y,0.3,marker='D',color=[0.3,0.3,0.4],alpha=0.05)
    if len(bin_spikes)==len(x):
        ax.scatter(x,y,s=bin_spikes, alpha=0.1, color = 'r')
    ax.set_axis_off()
    ax.set_xlim(TMF.x_limit)
    ax.set_ylim(TMF.y_limit)

    return ax

def plotZonesHeatMap(ax,cax,data,zones=TMF.ZonesNames,cmap='div',alpha=1,colArray=[]):
    if len(colArray)==0:
        if cmap=='div':
            cDat,colArray =  getDatColorMap(data)
            cMap = plt.get_cmap('RdBu_r')
        else:
            cDat,colArray =  getDatColorMap(data,col_palette='YlOrBr_r',div=False)
            cMap = plt.get_cmap('YlOrBr_r')
    else:
        if cmap=='div':
            cDat,_ =  getDatColorMap(data,colArray=colArray)
            cMap = plt.get_cmap('RdBu_r')
        else:
            cDat,_ =  getDatColorMap(data,colArray=colArray,col_palette='YlOrBr_r',div=False)
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

def getDatColorMap(data, nBins = 25, col_palette="RdBu_r",div=True,colArray=[]):

    if len(colArray)>0:
        nBins = len(colArray)
    else:
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
################################################################################
################################################################################
################################################################################
##### Stat annot code!, this is a copy of the statannot github. ################
################################################################################
################################################################################
################################################################################
def stat_test(box_data1, box_data2, test):
    testShortName = ''
    formattedOutput = None
    if test == 'Mann-Whitney':
        u_stat, pval = stats.mannwhitneyu(box_data1, box_data2, alternative='two-sided')
        testShortName = 'M.W.W.'
        formattedOutput = "MWW RankSum two-sided P_val={:.3e} U_stat={:.3e}".format(pval, u_stat)
    elif test == 't-test_ind':
        stat, pval = stats.ttest_ind(a=box_data1, b=box_data2)
        testShortName = 't-test_ind'
        formattedOutput = "t-test independent samples, P_val={:.3e} stat={:.3e}".format(pval, stat)
    elif test == 't-test_paired':
        stat, pval = stats.ttest_rel(a=box_data1, b=box_data2)
        testShortName = 't-test_rel'
        formattedOutput = "t-test paired samples, P_val={:.3e} stat={:.3e}".format(pval, stat)

    return pval, formattedOutput, testShortName



def pvalAnnotation_text(x, pvalueThresholds):
    singleValue = False
    if type(x) is np.array:
        x1 = x
    else:
        x1 = np.array([x])
        singleValue = True
    # Sort the threshold array
    pvalueThresholds = pd.DataFrame(pvalueThresholds).sort_values(by=0, ascending=False).values
    xAnnot = pd.Series(["" for _ in range(len(x1))])
    for i in range(0, len(pvalueThresholds)):
        if (i < len(pvalueThresholds)-1):
            condition = (x1 <= pvalueThresholds[i][0]) & (pvalueThresholds[i+1][0] < x1)
            xAnnot[condition] = pvalueThresholds[i][1]
        else:
            condition = x1 < pvalueThresholds[i][0]
            xAnnot[condition] = pvalueThresholds[i][1]

    return xAnnot if not singleValue else xAnnot.iloc[0]


def add_stat_annotation(ax,
                        data=None, x=None, y=None, hue=None, order=None, hue_order=None,
                        boxPairList=None,
                        test='Mann-Whitney', textFormat='star', loc='inside',
                        pvalueThresholds=[[1,"ns"], [0.05,"*"], [1e-2,"**"], [1e-3,"***"], [1e-4,"****"]],
                        useFixedOffset=False, lineYOffsetToBoxAxesCoord=None, lineYOffsetAxesCoord=None,
                        lineHeightAxesCoord=0.02, textYOffsetPoints=1,
                        color='0.2', linewidth=1.5, fontsize='medium', verbose=1):
    """
    User should use the same argument for the data, x, y, hue, order, hue_order as the seaborn boxplot function.

    boxPairList can be of either form:
    For non-grouped boxplot: [(cat1, cat2), (cat3, cat4)]
    For boxplot grouped by hue: [((cat1, hue1), (cat2, hue2)), ((cat3, hue3), (cat4, hue4))]
    """

    def find_x_position_box(boxPlotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")
        """
        if boxPlotter.plot_hues is None:
            cat = boxName
            hueOffset = 0
        else:
            cat = boxName[0]
            hue = boxName[1]
            hueOffset = boxPlotter.hue_offsets[boxPlotter.hue_names.index(hue)]

        groupPos = boxPlotter.group_names.index(cat)
        boxPos = groupPos + hueOffset
        return boxPos


    def get_box_data(boxPlotter, boxName):
        """
        boxName can be either a name "cat" or a tuple ("cat", "hue")

        Here we really have to duplicate seaborn code, because there is not direct access to the
        box_data in the BoxPlotter class.
        """
        if boxPlotter.plot_hues is None:
            cat = boxName
        else:
            cat = boxName[0]
            hue = boxName[1]

        i = boxPlotter.group_names.index(cat)
        group_data = boxPlotter.plot_data[i]

        if boxPlotter.plot_hues is None:
            # Draw a single box or a set of boxes
            # with a single level of grouping
            box_data = remove_na(group_data)
        else:
            hue_level = hue
            hue_mask = boxPlotter.plot_hues[i] == hue_level
            box_data = remove_na(group_data[hue_mask])

        return box_data

    fig = plt.gcf()

    validList = ['inside', 'outside']
    if loc not in validList:
        raise ValueError("loc value should be one of the following: {}.".format(', '.join(validList)))
    validList = ['t-test_ind', 't-test_paired', 'Mann-Whitney']
    if test not in validList:
        raise ValueError("test value should be one of the following: {}.".format(', '.join(validList)))

    if verbose >= 1 and textFormat == 'star':
        print("pvalue annotation legend:")
        pvalueThresholds = pd.DataFrame(pvalueThresholds).sort_values(by=0, ascending=False).values
        for i in range(0, len(pvalueThresholds)):
            if (i < len(pvalueThresholds)-1):
                print('{}: {:.2e} < p <= {:.2e}'.format(pvalueThresholds[i][1], pvalueThresholds[i+1][0], pvalueThresholds[i][0]))
            else:
                print('{}: p <= {:.2e}'.format(pvalueThresholds[i][1], pvalueThresholds[i][0]))
        print()

    # Create the same BoxPlotter object as seaborn's boxplot
    boxPlotter = sns.categorical._BoxPlotter(x, y, hue, data, order, hue_order,
                                             orient=None, width=.8, color=None, palette=None, saturation=.75,
                                             dodge=True, fliersize=5, linewidth=None)
    plotData = boxPlotter.plot_data

    xtickslabels = [t.get_text() for t in ax.xaxis.get_ticklabels()]
    ylim = ax.get_ylim()
    yRange = ylim[1] - ylim[0]

    if lineYOffsetAxesCoord is None:
        if loc == 'inside':
            lineYOffsetAxesCoord = 0.05
            if lineYOffsetToBoxAxesCoord is None:
                lineYOffsetToBoxAxesCoord = 0.06
        elif loc == 'outside':
            lineYOffsetAxesCoord = 0.03
            lineYOffsetToBoxAxesCoord = lineYOffsetAxesCoord
    else:
        if loc == 'inside':
            if lineYOffsetToBoxAxesCoord is None:
                lineYOffsetToBoxAxesCoord = 0.06
        elif loc == 'outside':
            lineYOffsetToBoxAxesCoord = lineYOffsetAxesCoord
    yOffset = lineYOffsetAxesCoord*yRange
    yOffsetToBox = lineYOffsetToBoxAxesCoord*yRange

    yStack = []
    annList = []
    for box1, box2 in boxPairList:

        valid = None
        groupNames = boxPlotter.group_names
        hueNames = boxPlotter.hue_names
        if boxPlotter.plot_hues is None:
            cat1 = box1
            cat2 = box2
            hue1 = None
            hue2 = None
            label1 = '{}'.format(cat1)
            label2 = '{}'.format(cat2)
            valid = cat1 in groupNames and cat2 in groupNames
        else:
            cat1 = box1[0]
            hue1 = box1[1]
            cat2 = box2[0]
            hue2 = box2[1]
            label1 = '{}_{}'.format(cat1, hue1)
            label2 = '{}_{}'.format(cat2, hue2)
            valid = cat1 in groupNames and cat2 in groupNames and hue1 in hueNames and hue2 in hueNames


        if valid:
            # Get position of boxes
            x1 = find_x_position_box(boxPlotter, box1)
            x2 = find_x_position_box(boxPlotter, box2)
            box_data1 = get_box_data(boxPlotter, box1)
            box_data2 = get_box_data(boxPlotter, box2)
            ymax1 = box_data1.max()
            ymax2 = box_data2.max()

            pval, formattedOutput, testShortName = stat_test(box_data1, box_data2, test)
            if verbose >= 2: print ("{} v.s. {}: {}".format(label1, label2, formattedOutput))

            if textFormat == 'full':
                text = "{} p < {:.2e}".format(testShortName, pval)
            elif textFormat is None:
                text = None
            elif textFormat is 'star':
                text = pvalAnnotation_text(pval, pvalueThresholds)

            if loc == 'inside':
                yRef = max(ymax1, ymax2)
            elif loc == 'outside':
                yRef = ylim[1]

            if len(yStack) > 0:
                yRef2 = max(yRef, max(yStack))
            else:
                yRef2 = yRef

            if len(yStack) == 0:
                y = yRef2 + yOffsetToBox
            else:
                y = yRef2 + yOffset
            h = lineHeightAxesCoord*yRange
            lineX, lineY = [x1, x1, x2, x2], [y, y + h, y + h, y]
            if loc == 'inside':
                ax.plot(lineX, lineY, lw=linewidth, c=color)
            elif loc == 'outside':
                line = lines.Line2D(lineX, lineY, lw=linewidth, c=color, transform=ax.transData)
                line.set_clip_on(False)
                ax.add_line(line)

            if text is not None:
                ann = ax.annotate(text, xy=(np.mean([x1, x2]), y + h),
                                  xytext=(0, textYOffsetPoints), textcoords='offset points',
                                  xycoords='data', ha='center', va='bottom', fontsize=fontsize,
                                  clip_on=False, annotation_clip=False)
                annList.append(ann)

            ax.set_ylim((ylim[0], 1.1*(y + h)))

            if text is not None:
                plt.draw()
                yTopAnnot = None
                gotMatplotlibError = False
                if not useFixedOffset:
                    try:
                        bbox = ann.get_window_extent()
                        bbox_data = bbox.transformed(ax.transData.inverted())
                        yTopAnnot = bbox_data.ymax
                    except RuntimeError:
                        gotMatplotlibError = True

                if useFixedOffset or gotMatplotlibError:
                    if verbose >= 1:
                        print("Warning: cannot get the text bounding box. Falling back to a fixed y offset. Layout may be not optimal.")
                    # We will apply a fixed offset in points, based on the font size of the annotation.
                    fontsizePoints = FontProperties(size='medium').get_size_in_points()
                    offsetTrans = mtransforms.offset_copy(ax.transData, fig=fig,
                                                          x=0, y=1.0*fontsizePoints + textYOffsetPoints, units='points')
                    yTopDisplay = offsetTrans.transform((0, y + h))
                    yTopAnnot = ax.transData.inverted().transform(yTopDisplay)[1]
            else:
                yTopAnnot = y + h

            yStack.append(yTopAnnot)
        else:
            raise ValueError("boxPairList contains an unvalid box pair.")
            pass


    yStackMax = max(yStack)
    if loc == 'inside':
        ax.set_ylim((ylim[0], 1.03*yStackMax))
    elif loc == 'outside':
        ax.set_ylim((ylim[0], ylim[1]))

    return ax
