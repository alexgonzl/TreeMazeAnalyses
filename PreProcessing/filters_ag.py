import numpy as np
from scipy.interpolate import interp1d

def medFilt(x,window):
    ''' moving median filter that can take np.nan as entries.
        note that the filter is non-causal, output of sample ii is the median
        of samples of the corresponding window centered around ii.
    Inputs:
        x       ->  signal to filtered,
        window  ->  number of samples to use for median estimation.

    Output:
        y       <-  median filtered signal
    '''
    if window%2:
            window=window-1
    win2 = np.int(window/2)
    N=len(x)
    y=np.array(x)
    for ii in np.arange(win2,N-win2+1):
        try:
            idx=(np.arange(-win2,win2)+ii).astype(np.int)
            y[ii] = np.nanmedian(y[idx])
        except:
            pass
    return y


def medFiltFilt(x,window):
    ''' moving median filter that can take np.nan as entries.
        note that the filter is non-causal, output of sample ii is the median
        of samples of the corresponding window centered around ii. This repeats
        function repeats the median operation in the reverse direction.
    Inputs:
        x       ->  signal to filtered,
        window  ->  number of samples to use for median estimation.

    Output:
        y       <-  median filtered signal
    '''
    if window%2:
            window=window-1
    win2 = np.int(window/2)
    N=len(x)
    y=np.array(x)
    for ii in np.arange(win2,N-win2+1):
        try:
            idx=(np.arange(-win2,win2)+ii).astype(np.int)
            y[ii] = np.nanmedian(y[idx])
        except:
            pass
    for ii in np.arange(N-win2,win2-1,-1):
        try:
            idx=(np.arange(-win2,win2)+ii).astype(np.int)
            y[ii] = np.nanmedian(y[idx])
        except:
            pass
    return y

def ReSampleDat(t,sig,step):
    '''Nearest neighbor interpolation for resampling.
    Inputs:
        t    ->  original time series
        sig  ->  signal to be resample
        step ->  time step for resampleing (e.g step = 0.02 -> 20ms)

    Output:
        tp   <-  new time series created with step
        out  <-  resampled signal
    e.g.
        tp,xs = ReSampleDat(t,x,0.02)
    '''

    tp = np.arange(t[0],t[-1],step)
    sig_ip = interp1d(t, sig, kind="nearest",fill_value="extrapolate")
    out = sig_ip(tp)
    return tp, out
