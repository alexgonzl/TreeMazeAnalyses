import numpy as np
import matplotlib as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
import statsmodels.stats.api as sms
import scipy.stats as sps
import sys
sys.path.append('../PreProcessing/')
sys.path.append('../TrackingAnalyses/')
sys.path.append('../Lib/')
sys.path.append('../misc/')
sys.path.append('../Analyses/')
import TreeMazeFunctions as TMF

def spatial_information(loc_prob, fr_map):
    meanFR = np.nanmean(fr_map)
    InfoMat = fr_map*loc_prob*np.log2(fr_map/meanFR)
    return 1/meanFR*np.nansum(InfoMat)

def getRandomPerm(x,seed=0):
    np.random.seed(seed)
    return np.random.permutation(x)

def BiPermTest(func, x, y, n=1000,seed=0):
    out = np.zeros(n)
    np.random.seed(seed)
    for i in np.arange(n):
        x2 = np.random.permutation(x)
        out[i] = func(x2,y)
    return out
def SI_Zone_Perm(ZoneInfo,zones, fr ,n=1000,seed=0):
    si_sh = np.zeros(n)
    np.random.seed(seed)
    zfr = np.bincount(zones,fr,minlength=TMF.nZones)/ZoneInfo.loc['counts']
    si = spatial_information(ZoneInfo.loc['prob'],zfr)
    for i in np.arange(n):
        x2 = np.random.permutation(zones)
        zfr_sh = np.bincount(x2,fr,minlength=TMF.nZones)/ZoneInfo.loc['counts']
        si_sh[i] = spatial_information(ZoneInfo.loc['prob'],zfr_sh)
    p = 1-np.sum(si>si_sh)/n
    if p==0:
        p = 1/(n+1)
    elif p==1:
        p=1-1/(n+1)
    return si,si_sh,p
def SIPermTest(x,y,n=1000,seed=0):
    z = spatial_information(x,y)
    out = BiPermTest(spatial_information,x,y,n,seed)
    p = 1-np.sum(z>out)/n
    if p==0:
        p = 1/(n+1)
    elif p==1:
        p=1-1/(n+1)
    return z,out,p

def plotPermDist(ax,dist,trueVal,pval):
    with sns.plotting_context('notebook'):
        sns.set(style="white", palette="muted",font_scale=1.2)
        sns.despine(left=True)

        _=sns.distplot(dist,bins=25,ax=ax)
        ax.axvline(trueVal,color='r',linewidth=2,alpha=0.8,linestyle='--')
        ax.add_artist(AnchoredText('si={0:.3f}\np={1:.3f}'.format(trueVal,pval),loc='upper right',frameon=False))
        _=ax.set_xlabel('Spatial Information')
        _=ax.set_ylabel('')
        ax.set_yticks([])
    return ax

def PDeviance(y,mu):
    y = y+1e-10
    mu = mu+1e-10
    return 2*np.sum(y*np.log(y/mu)-(y-mu))

def getSpikeStats(data, groups):

    groupIDs = np.unique(groups).astype(int)
    nGroups = len(groupIDs)
    stats = {}
    stats['mean'] = np.zeros(nGroups)
    stats['sem'] = np.zeros(nGroups)
    stats['conf_Int'] = np.zeros((nGroups,2))
    stats['N'] = np.zeros(nGroups)
    stats['MWUz'] = np.zeros(nGroups)

    for i in groupIDs:
        g1 = groups==i
        g2 = groups!=i
        x_stats = sms.DescrStatsW(data[g1])
        stats['mean'][i] = x_stats.mean
        stats['sem'][i] = x_stats.std_mean
        stats['N'][i] = x_stats.nobs
        stats['conf_Int'][i] = x_stats.tconfint_mean()
        stats['MWUz'][i],_,_ = getMWUz(data,g1,g2)

    return stats

def getMWUz(data,g1,g2):
    n1 = np.sum(g1)
    n2 = np.sum(g2)
    mu = n1*n2/2
    su = np.sqrt( n1*n2*(n1+n2+1)/12)
    u,p=sps.mannwhitneyu(data[g1],data[g2])
    t,_=sps.ttest_ind(data[g1],data[g2])
    zu = (u-mu)/su
    zu2=np.sign(t)*np.abs(zu)
    return zu2, p, u
