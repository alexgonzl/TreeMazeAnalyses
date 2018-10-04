###############################################################################
###############################################################################
# get_neuralynx.py
# Collection of routines for reading neuralynx file formats. Core functionality
# exported from the nept python package.
#
# Dependencies:
# nept
#
# written by Alex Gonzalez
# last edited 10/4/18
# Stanford University
# Giocomo Lab
###############################################################################
###############################################################################

import numpy as np
from scipy import signal
import nept
import h5py

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

    ev=nept.load_events(fn,events)
    return ev

def get_save_events(ev_files,sp):
    for ev_file in ev_files:
        ev = get_events(ev_file[2])
        with h5py.File(str(sp / ev_file[1])+'.h5', 'w') as hf:
            for k,v in ev.items():
                hf.create_dataset(k,  data=v)

def get_save_tracking(vt_files,sp):
    for vt_file in vt_files:
        t,x,y = get_position(vt_file[2])
        with h5py.File(str(sp / vt_file[1])+'.h5', 'w') as hf:
            hf.create_dataset("t",  data=t)
            hf.create_dataset("x",  data=x)
            hf.create_dataset("y",  data=y)
