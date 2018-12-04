
import numpy as np
from pathlib import Path
import json

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

def saveNormFactor(factor,sp):
    x = {}
    x['Normalization Factor '] = factor
    with open(str(save_path/'header_tt')+str(tid)+'.json', 'w') as f:
        json.dump(header, f ,indent=4)
    pass

def npy2bin(filename,savepath,overwriteFlag=0):
    file = Path(filename)
    sp = Path(savepath)
    if file.exists():
        if file.suffix == '.npy':
            if not (sp / str(file.stem +'.bin')).exists() or overwriteFlag:
                try:
                    data = np.load(file)
                    normFactor = getInt16ConvFactor(data)
                    dat2 = data2int16(data,normFactor)
                    dat2.tofile(sp / str(file.stem +'.bin'))
                    saveNormFactor(normFactor)
                except:
                    print('Error procesing {}'.format(file))
                    print(sys.exc_info()[0],sys.exc_info()[1],sys.exc_info()[2].tb_lineno)
                    sys.exit()
            print('File exists and overwrite = false ')        
        else:
            sys.exit('Invalid File')
    else:
        sys.exist('File Not Found {}'.format(file))
