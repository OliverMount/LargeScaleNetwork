
#--------------------------------------------------------------	
# Program to compute and store the transfer entropy for the Deco fMRI time series
# This program is using the IDTxl tool box  
#-------------------------------------------------------------- 

import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.io import loadmat 
import mat73
import pandas as pd
import seaborn as sns


from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network
import time

def min_max(T):
	mi=T.min()
	ma=T.max()
	return (T-mi)/(ma-mi)


dpath='/olive/Maths/R/Do/InputData/'

sli=np.hstack((range(34),range(41,75)))
## Loading the fMRI data path
data=loadmat(dpath+"DKatlas_timeseries.mat")
dat=data['ts']
data=dat[:,sli,:]
print("Shape of the HCP data from Deco. is :", dat.shape)
print("Shape of the HCP data after ROI selection is :", data.shape)

# NoT X nROI X NoS
#plt.plot(data[:,4,0])
#plt.show()
print("\n")

with open(dpath+"Filtered.npy",'rb') as f:
    fdata= np.load(f)          # Filtered data loading
    print("Done with loading the filtered data")

sha=fdata.shape
L=sha[0]
N_ROI=sha[1]
NoS=sha[2]

print("Shape of the filtered data is ", sha)
print("Total number of participants considered is " ,NoS)
print("Total number of ROIs considered is " ,N_ROI)
print("Length of the time series is " ,L)


def get_TEmtx(results,fdr=True,nori=N_ROI):

    TEmtx=np.zeros((N_ROI,N_ROI))

    for tgt in range(N_ROI):
        temp=results.get_single_target(tgt,fdr)
        src_idx= np.unique([k[0] for k in temp['selected_vars_sources']])
        TEvals=temp['te']
        L=len(src_idx)
        for k in range(L):
            TEmtx[src_idx[k],tgt]=TEvals[k]
    return TEmtx

def summary_TE(TEmtx):
    print('TE matrix is:')
    print(TEmtx)
    print('Source values')
    print(TEmtx.mean(axis=1))

    print('Target values')
    print(TEmtx.mean(axis=0))

    print('Source-Target values')
    SrmTg=TEmtx.mean(axis=1)-TEmtx.mean(axis=0)
    print(SrmTg)

    Eff_sources=sum(SrmTg>0)
    Eff_targets=sum(SrmTg<=0)

    print('No. of Effective sources is :', Eff_sources)
    print('No. of Effective targets is :', Eff_targets)


max_delay=[10]   # History in terms of TRs (Deco. caluclated via the decay of Autocorrelation, max of 10 TR)
L=len(max_delay)

TE=np.zeros((N_ROI,N_ROI,L,NoS))
Delay=np.zeros((N_ROI,N_ROI,L,NoS))

for kp in range(NoS):
	
	st_p=time.time()
	# a) Loading the data
	print('++ Begin computing TE for participant-'+str(kp+1)+' using filtered fMRI data')
	data_te = Data(data[:,:N_ROI,kp],'sp')

	#b) Initialise analysis object and define settings
	
	network_analysis=BivariateTE()

	settings={'cmi_estimator': 'JidtGaussianCMI',
        'max_lag_sources': max_delay[0], 
        'min_lag_sources': 1,
        'verbose':False}

	#c) Run analsis
	results=network_analysis.analyse_network(settings=settings,data=data_te)

	#d) Obtain the TE and delay matrices from the network result and store them  
	
	TEmtx=get_TEmtx(results,False,N_ROI)     # To obtain the TE matrix
	summary_TE(TEmtx)
	M=results.get_adjacency_matrix('max_te_lag',False).get_matrix()

    #plt.imshow(TEmtx,cmap='jet');plt.colorbar();plt.show()
    #plt.imshow(M,cmap='jet');plt.colorbar();plt.show()
	TE[:,:,0,kp]=TEmtx     # The index 0 here is for history extension in the future
	Delay[:,:,0,kp]=M
	print('++ Computed TE for participant-'+str(kp+1) + ' in ' +str(time.time()-st_p)+ ' sec')

print('++ Saving the Transfer entropy  and the delay file')
np.save('TE_fMRI_unfilt.npy',TE)
