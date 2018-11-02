###############################################################################
###############################################################################
# pre_process_tetrode.py
# Collection of routines and subroutines for processing raw neuralynx CSC files.
#
# Dependencies:
# get_neuralynx.py
# robust_stats.py
#
# written by Alex Gonzalez
# last edited 10/4/18
# Stanford University
# Giocomo Lab
###############################################################################
###############################################################################
import numpy as np
import time
from scipy import signal
from robust_stats import *
from get_neuralynx import get_csc, get_header
import re
import pandas as pd
import csv
import sys
import h5py

def get_process_save_tetrode(tetrode_files, save_path, save_format='npy', AmpPercentileThr=0.975):
    ''' This function takes a list of neuralynx continously sampled channel (csc) files,
     performs initial preprocessing, data QA, and saves all the channels from a tetrode together.
     The files should come from the same recording, have the same length, and follow the naming
     format. The naming format is as follows:
     Tetrode 1 files -> CSC1a.csc, CSC1b.csc, CSC1c.csc, CSC1d.csc

     This function can be used in parallel by having the csc_list only include a
     subset of tetrodes, and calling it multiple times to complete the set.
     If there are 16 tetrodes, it can be processed with 16 streams.

     Parameters
    ----------
    csc_files : list of data file names (CSCs)
        this is a list of arrays containing: filename, filename_stem, file_path,
        and last modification time.
        e.g. csc_files[0]=['csc1a.csc', 'csc1a', '~/mydata', 'date in some format' ]
    save_path: string or path object
        full path for data to be saved
    save_format: string ('npy,'csv' or 'h5'); default npy
        currently supporting numpy arrays (.npy), comma separeted values (csv), or h5

    AmpPercentileThr: double [0-1], default = 0.975
        Threshould for amplitude rejection based on maximum value (clipping).
        example: AmpPercentileThr=0.9, and max value (from meta data) = 1000
        implies that for signal 's' values exceed 900 are set to zero:
        in code... s[abs(s)>= 900]=0

    Output:
    ----------
    This function processes a list of csc files, and saves them in sets of 4 (tetrodes).
    It won't explicitely output anything.

    Output Files.
    - processed tetrodes
    - header info and processing details
    - timestap file (times markers for the number of samples of the recording)

    '''

    # Filter Parameters (must excist in the same directory)
    try:
        b = np.fromfile('filt_b.dat',dtype='float',sep=',')
        a = np.fromfile('filt_a.dat',dtype='float',sep=',')
    except:
        error('Filters not found in the working directory. Aborting.')

    id1 = np.where((df["tetrodeIDs"]== 1) & (df["tetrodeChanIDs"] == 'a'))[0][0]
    h1  = get_header(df.loc[id1].full_file_path)
    sig,time_stamps = get_csc(df.loc[id1].full_file_path)
    nSamps = len(sig)
    del sig

    # save time stamps
    save_timestamps(time_stamps, save_path, save_option)

    for tetrode_id in tetrodeUniqueIDs:
        # temp = 'tt_{}.npy'.format(tetrode_id)
        # if not (sp / temp).exists():
        info  = {'tetrodeID':[tetrode_id],'RefChan':h1['RefChan'],
        'fs':h1['fs'],'AmpPercentileThr':AmpPercentileThr,'nSamps':nSamps}
        tetrode = np.zeros((nSamps,4),dtype=np.float32)
        cnt =0

        if not (sp / 'tt_{}.npy'.format(tetrode_id)).exists():
            for chan_id in ['a','b','c','d']:
                unique_id = np.where((df.tetrodeIDs== tetrode_id) & (df.tetrodeChanIDs == chan_id))[0][0]

                fn = df.full_file_path[unique_id]

                print("\nProcessing Tetrode {} Channel {}".format(tetrode_id,chan_id))
                # Load signal
                sig,temp = get_csc(fn)
                del temp

                h2  = get_header(fn)
                info['AD'+'_'+chan_id]=h2['AD']
                info['InputRange'+'_'+chan_id]=h2['InputRange']
                info['AmpRejThr'+'_'+chan_id] = h2['InputRange']*AmpPercentileThr
                rejThr= h2['InputRange']*AmpPercentileThr

                ## Step 1. Filter.
                t1=time.time()
                fsig = FilterCSC(sig,b,a)
                t2=time.time()
                print("Time to filter the signal %0.2f" % (t2-t1))

                ## Step 2. Artifact reject ceiling/clipping events
                fsig,nBadSamps = AmpRej(fsig,sig,rejThr)
                print("Number of rejected samples {}".format(nBadSamps))
                t3=time.time()
                print("Time to reject bad signal segments %0.2f" % (t3-t2))

                ## Step 3. Standarized the signal
                fsig = robust_zscore(fsig)
                t4 = time.time()
                print("Time to normalize signal %0.2f" % (t4-t3))

                # Overall time and return
                print("Total time to preprocess the signal %0.2f" % (t4-t1))

                tetrode[:,cnt]=fsig
                info['nBadAmpSamps'+'_'+chan_id]=nBadSamps
                cnt=cnt+1
            save_tetrode(tetrode,save_path,tetrode_id,save_format)
            save_tetrode_info(info,tetrode_id,save_path)


######################## Auxiliary Functions ##########################
def save_tetrode(tetrode,save_path,tid,save_format):
    if save_format=='h5':
        with h5py.File(str(save_path / 'tt_')+str(tid)+'.h5', 'w') as hf:
            hf.create_dataset("tetrode",  data=tetrode)
    elif save_format=='npy':
        np.save(str(save_path / 'tt_')+str(tid),tetrode)
    elif save_format=='csv':
        np.savetxt(str(save_path / 'tt_')+str(tid)+'.csv',tetrode,delimiter=',')
    print('Tetrode {} reuslts saved to '.format(tid)  + str(save_path))

def save_tetrode_info(header,tid,save_path):
    with open(str(save_path / 'header_tt')+str(tid),'w') as f:
        w = csv.writer(f)
        for key, value in header.items():
            print(key,value)
            w.writerow([key, value])

def save_timestamps(stamps, save_path, save_format):
    if save_format=='h5':
        with h5py.File(str(save_path / 'time_stamps')+'.h5', 'w') as hf:
            hf.create_dataset("time_stamps",  data=stamps)
    elif save_format=='npy':
        np.save(str(save_path / 'time_stamps'),stamps)
    elif save_format=='csv':
        np.savetxt(str(save_path / 'time_stamps')+'.csv',stamps,delimiter=',')

def FilterCSC(fsig,b,a):
    return signal.filtfilt(b,a,fsig)

def AmpRej(fsig,sig,thr):
    badSamps= np.where(abs(sig)>=thr)[0]
    fsig[badSamps]=0
    return fsig, np.int(len(badSamps))
