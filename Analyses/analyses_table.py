# creates a table with progress of analyses for each session
# update analyses table

from pathlib import Path
import os,sys, json, datetime, getopt
import pickle as pkl
import time

oakPaths = {}
oakPaths['Root'] = Path('/mnt/o/giocomo/alexg/')
oakPaths['Clustered'] = Path('/mnt/o/giocomo/alexg/Clustered/')
oakPaths['PreProcessed'] = Path('/mnt/o/giocomo/alexg/PreProcessed/')
oakPaths['Raw'] = Path('/mnt/o/giocomo/alexg/RawData/InVivo/')
oakPaths['Analyses'] = Path('/mnt/o/giocomo/alexg/Analyses')

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

if __name__ == '__main__':
    ID = ''
    minFileSize = 16384
    TetrodeRecording = 1
    nTetrodes = 16

    if len(sys.argv)<2:
        print("Usage: %s -a ID " % sys.argv[0])
        sys.exit('Invalid input.')

    print(sys.argv[1:])
    myopts, args = getopt.getopt(sys.argv[1:],"a:p:")
    for o, a in myopts:
        print(o,a)
        if o == '-a':
            ID = str(a)
        elif o == '-p':
            if str(a)=='NR32':
                TetrodeRecording = 0
                nChannels = 32
            elif str(a)=='TT16':
                TetrodeRecording = 1
                nTetrodes=16
            else:
                sys.exit('Invalid Probe Type.')
        else:
            print("Usage: %s -a ID " % sys.argv[0])
            sys.exit('Invalid input. Aborting.')
