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
import time, datetime
from scipy import signal
from robust_stats import *
from pathlib import Path
import csv
import sys
import h5py, json
import nept

def get_process_save_tetrode(task, save_format='bin', AmpPercentileThr=0.975, overwriteFlag=0):
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

    # Unpack the tetrode information.
    tt_id = task['tt_id']
    csc_files = task['filenames']
    sp = Path(task['sp'])
    if not (sp / 'tt_{}.{}'.format(tt_id,save_format)).exists() or overwriteFlag :
        # Filter Parameters (must excist in the same directory)
        try:
            b = np.fromfile('filt_b.dat',dtype='float',sep=',')
            a = np.fromfile('filt_a.dat',dtype='float',sep=',')
        except:
            sys.exit('Filters not found in the working directory. Aborting.')

        date_obj = datetime.date.today()
        date_str= "%s_%s_%s" % (date_obj.month,date_obj.day,date_obj.year)

        # load first channel
        f =[]
        f.append(csc_files[0])
        try:
            h1  = get_header(f[0])
            sig,time_stamps = get_csc(f[0])
            nSamps = len(sig)
        except:
            sys.exit('Could not read first channel. Aborting.')

        # save time stamps
        save_timestamps(time_stamps, sp, save_format)
        del time_stamps

        # create information file
        info  = {'tetrodeID':[tt_id],'RefChan':h1['RefChan'],
        'fs':h1['fs'],'AmpPercentileThr':AmpPercentileThr,'nSamps':nSamps, 'date_processed': date_str}

        data = np.zeros((nSamps,4),dtype=np.float32)
        cnt =0
        for chan_id in np.arange(0,4):
            print("\nProcessing Tetrode {} Channel {}".format(tt_id,chan_id))
            # load channel data
            if chan_id>0:
                f.append(csc_files[chan_id])
                # Load signal
                sig,temp = get_csc(f[chan_id])
                del temp
            chan_id_str = str(chan_id)

            # get channel specific info
            h2  = get_header(f[chan_id])
            info['AD'+'_'+chan_id_str]=float(h2['AD'])
            info['InputRange'+'_'+chan_id_str]=h2['InputRange']
            info['AmpRejThr'+'_'+chan_id_str] = h2['InputRange']*AmpPercentileThr
            info['date_created'+'_'+chan_id_str]=get_file_date(f[chan_id])
            rejThr= h2['InputRange']*AmpPercentileThr

            ## Step 1. Filter.
            t1=time.time()
            fsig = FilterCSC(sig,b,a)
            fsig=sig
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

            data[:,cnt]=fsig
            info['nBadAmpSamps_'+chan_id_str]=nBadSamps

            cnt=cnt+1
            print('Processing TT {} Channel {} completed.\n'.format(tt_id,chan_id))

        if save_format=='bin':
            int16NormFactor = getInt16ConvFactor(data)
            info['Int_16_Norm_Factors = '] = int16NormFactor
            save_tetrode(data,sp,tt_id,save_format,int16NormFactor)
        else:
            save_tetrode(data,sp,tt_id,save_format)
        save_tetrode_info(info,tt_id,sp)
    else:
        print('File exists and overwrite = false ')

def get_save_events(task,overwriteFlag=0):
    ev_file = task['filenames']
    sp = Path(task['sp'])

    if not (sp / 'ev.h5').exists() or overwriteFlag:
        ev = get_events(ev_file)
        with h5py.File(str(sp / 'ev')+'.h5', 'w') as hf:
            for k,v in ev.items():
                hf.create_dataset(k,  data=v)
    else:
        print('File exists and overwrite = false ')

def get_save_tracking(task,overwriteFlag=0):
    vt_file = task['filenames']
    sp = Path(task['sp'])
    if not (sp / 'vt.h5').exists() or overwriteFlag :
        t,x,y = get_position(vt_file)
        with h5py.File(str(sp / 'vt')+'.h5', 'w') as hf:
            hf.create_dataset("t",  data=t)
            hf.create_dataset("x",  data=x)
            hf.create_dataset("y",  data=y)
    else:
        print('File exists and overwrite = false ')

################################################################################
################################################################################
######################## Auxiliary Functions ###################################
################################################################################
################################################################################

def save_tetrode(tetrode,save_path,tid,save_format,int16NormFactor=1):
    if save_format=='h5': # h5 format
        with h5py.File(str(save_path / 'tt_')+str(tid)+'.h5', 'w') as hf:
            hf.create_dataset("tetrode",  data=tetrode)
    elif save_format=='npy': # numpy
        np.save(str(save_path / 'tt_')+str(tid),tetrode)
    elif save_format=='csv': # comma separeted values
        np.savetxt(str(save_path / 'tt_')+str(tid)+'.csv',tetrode,delimiter=',')
    elif save_format=='bin': # binary
        data=data2int16(tetrode,int16NormFactor)
        data.tofile(str(save_path / 'tt_')+str(tid)+'.bin')
    else:
        print('Unsuported save method specified {}, saving as .npy array.'.format(save_format))
        np.save(str(save_path / 'tt_')+str(tid),tetrode)

    print('Tetrode {} results saved to '.format(tid)  + str(save_path))
    print('')

def save_tetrode_info(header,tid,save_path):
    # with open(str(save_path / 'header_tt')+str(tid),'w') as f:
    #     w = csv.writer(f)
    #     for key, value in header.items():
    #         print(key,value)
    #         w.writerow([key, value])
    with open(str(save_path/'header_tt')+str(tid)+'.json', 'w') as f:
        json.dump(header, f ,indent=4)

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

def get_file_date(fn):
    fn = Path(fn)
    info=fn.stat()
    ymd=time.localtime(info.st_mtime)[0:3]
    return "%s_%s_%s" % (ymd[1],ymd[2],ymd[0])

def get_csc(fn):
    ''' returns signal in uV and time stamps'''
    temp = nept.load_lfp(fn)
    return np.float32(temp.data.flatten()*1e6), temp.time

def get_header(fn):
    h=nept.load_neuralynx_header(fn)
    for line in h.split(b'\n'):
        if line.strip().startswith(b'-ADBitVolts'):
            try:
                AD = np.array(float(line.split(b' ')[1].decode()))
            except ValueError:
                AD  = 1
        if line.strip().startswith(b'-ReferenceChannel'):
            try:
                RefChan = line.split(b' ')[3].decode()
                RefChan= int(RefChan[:-2])
            except ValueError:
                refChan=-1
        if line.strip().startswith(b'-SamplingFrequency'):
            try:
                fs = int(line.split(b' ')[1].decode())
            except ValueError:
                fs=32000
        if line.strip().startswith(b'-ADChannel'):
            try:
                ChanID = int(line.split(b' ')[1].decode())
            except ValueError:
                ChanID = -1
        if line.strip().startswith(b'-InputRange'):
            try:
                InputRange = int(line.split(b' ')[1].decode())
            except ValueError:
                InputRange = -1
    header  = {'AD': AD,'RefChan':RefChan,'fs':fs,'ChanID':ChanID,'InputRange':InputRange}
    return header

def get_position(fn):
    pos = nept.load_nvt(fn)
    #x=signal.medfilt(pos['x']/2.48 ,15)
    #y=signal.medfilt(pos['y']/2.48,15)
    t=pos['time']
    x=pos['x']
    y=pos['y']
    return t,x,y

def get_events(fn):
    events = {'DE1':'DE1','DE2':'DE2','DE3':'DE3','DE4':'DE4','DE5':'DE5','DE6':'DE6',
      'L1':'L1','L2':'L2','L3':'L3','L4':'L4','L5':'L5','L6':'L6',
      'RD':'RD','CL':'CL','CR':'CR'}

def data2int16(data,int16NormFactor):
    nChannels = data.shape[1] # 1 is the dim of channels
    data2=np.zeros(np.shape(data),np.int16)
    for ch in np.arange(nChannels):
        data2[:,ch] = np.floor(data[:,ch]*int16NormFactor)
    return data2

def getInt16ConvFactor(data):
    maxInt16Val = 32767
    minDatVal = np.min(data,0)
    maxDatVal = np.max(data,0)
    return maxInt16Val/np.maximum(abs(minDatVal),abs(maxDatVal))
