def atan2v(x,y):
    N = len(y)
    out = np.zeros(N)
    for i in np.arange(N):
        out[i] = np.math.atan2(y[i],x[i])
    return out

def velocity(x,y,window = 15):

    dx = np.diff(x)
    dy = np.diff(y)
    b = signal.firwin(window, cutoff = 0.2, window = "hanning")
    dx = signal.filtfilt(b,1,dx)
    dy = signal.filtfilt(b,1,dy)
    mag = euclidian(dx,dy)
    ang = atan2v(dx,dy)
    return mag , ang

def euclidian(x,y):
     return np.sqrt(x*x+y*y)
