import numpy as np
import pandas as pd
from scipy import signal, ndimage, interpolate, stats
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

import TreeMazeFunctions as TMF
import spike_functions as SF
import zone_analyses_session as ZA

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import AnchoredText

import nept
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
import TrialAnalyses as TA

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
    Paths['ClusterTable'] = rootPath['Clustered'] / animal / (animal+'_ClusteringSummary.json')
    Paths['Analyses'] = rootPath['Analyses'] / animal/ (session + '_Analyses')

    if not Paths['Clusters'].exists():
        print('Error, no Cluster Folder found.')
    if not Paths['PreProcessed'].exists():
        print('Error, no processed binaries found.')
    if not Paths['ClusterTable'].exists():
        print('Error, no clustering table found.')

    Paths['Analyses'].mkdir(parents=True, exist_ok=True)

    Paths['BehavTrackDat'] = Paths['Analyses'] / ('BehTrackVariables_{}ms.h5'.format(int(step*1000)))

    Paths['Cell_Spikes'] = Paths['Analyses'] / 'Cell_Spikes.json'
    Paths['Cell_Bin_Spikes'] = Paths['Analyses'] / ('Cell_Bin_Spikes_{}ms.npy'.format(int(step*1000)))
    Paths['Cell_FR'] = Paths['Analyses'] / ('Cell_FR_{}ms.npy'.format(int(step*1000)))

    Paths['Mua_Spikes'] = Paths['Analyses'] / 'Mua_Spikes.json'
    Paths['Mua_Bin_Spikes'] = Paths['Analyses'] / ('Mua_Bin_Spikes_{}ms.npy'.format(int(step*1000)))
    Paths['Mua_FR'] = Paths['Analyses'] / ('Mua_FR_{}ms.npy'.format(int(step*1000)))

    Paths['Spike_IDs'] = Paths['Analyses'] / 'Spike_IDs.json'
    Paths['ZoneAnalyses'] = Paths['Analyses'] / 'ZoneAnalyses.pkl'

    Paths['TrialInfo'] = Paths['Analyses'] / 'TrInfo.pkl'
    Paths['TrialCondMat'] = Paths['Analyses'] / 'TrialCondMat.csv'
    Paths['TrLongPosMat'] = Paths['Analyses'] / 'TrLongPosMat.csv'
    Paths['TrLongPosFRDat'] = Paths['Analyses'] / 'TrLongPosFRDat.csv'
    Paths['TrModelFits'] = Paths['Analyses'] /  'TrModelFits.csv'
    
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


    return Paths
