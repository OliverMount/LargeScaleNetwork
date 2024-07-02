
#--------------------------------------------------------------	
# Program to compute and store the transfer entropy for the Deco fMRI time series
#-------------------------------------------------------------- 

import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.io import loadmat 
import mat73

def min_max(T):
	mi=T.min()
	ma=T.max()
	return (T-mi)/(ma-mi)

from pyinform import transfer_entropy as te
import pandas as pd
import seaborn as sns

N_ROI=68
NoS=389

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
print("Shape of the filtered data is ", sha)

print("Total number of participants considered is " ,sha[2])

# sha=(68,68,389)
Lp=sha[2]            # No. of participants
history_points=[10]   # History in terms of TRs (Deco. caluclated via the decay of Autocorrelation, max of 10 TR)
L=len(history_points)

TE=np.zeros((N_ROI,N_ROI,L,Lp))
#plist=list(range(1,NoS+1))

for p in range(Lp):
	
	print('++ Dealing with participant-'+str(p+1)+' filtered fMRI data')
	for source in range(N_ROI):
		#print("++++ Determining TE of Source ROI-",source)
		for target in range(N_ROI):
			#print("++++++ Source: " ,source,"  Target: ",target)
			kh=0
			for k in history_points:
				TE[source,target,kh,p]=te(min_max(data[:,source,p]),min_max(data[:,target,p]),k)
				kh=kh+1	
	print('++ Computed TE for participant-'+str(p+1))

print("++ Saving the Transfer entropy file")
np.save("TE_fMRI_filt.npy",TE)
