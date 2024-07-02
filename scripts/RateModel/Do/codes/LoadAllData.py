#def LoadAllData():

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter,filtfilt  # For temporal filtering
from scipy.stats import pearsonr				  # For calulating r, and pvalue
import scipy.io as read   # for reading matlab files
import h5py
import mat73 as mat

dpath="/olive/Maths/R/Do/InputData/"
#-------------------------------------------------------------- 
# Loading data
#-------------------------------------------------------------- 

nROI=68    # Number of ROI
NoS=389	   # Number of participants	
TR=0.754   # [s] HCP data TR
NoT=616   # Length of time series data
#--------------------------------------------------------------
# Gene data loading and processing
#--------------------------------------------------------------


Genes_selected=read.loadmat(dpath+"DKcortex_selectedGenes.mat")
#print(Genes_selected.keys())

# 27 genes and 34 ROI
gene= Genes_selected['expMeasures']
print("Shape of gene data: ", gene.shape)  # 34 X 27

ratioE = np.zeros((nROI))
Coef_E=np.sum(gene[:,17:25],axis=1)     # 18:21 ampa+ 22:25 nmda/gaba
ratioE[:34]=Coef_E/Coef_E.max()
ratioE[34:]=ratioE[:34]                 # Assuming symmetry

ratioI = np.zeros((nROI))
sli=np.hstack((range(1,9),range(11,14)))
Coef_I=np.sum(gene[:,sli],axis=1)     # 18:21 ampa+ 22:25 nmda/gaba
ratioI[:34]=Coef_I/Coef_I.max()
ratioI[34:]=ratioI[:34]                 # Assuming symmetry

ratio=ratioE/ratioI
ratio=ratio/(max(ratio)-min(ratio))
ratio=ratio-max(ratio)+1                 # This value is verified

#print(ratio)
# Meylin data
Myelin=read.loadmat(dpath+"myelin_HCP_dk68.mat")
print("Shape of myelin data:",Myelin['t1t2Cortex'].shape)

#--------------------------------------------------------------
# Structural connectivity data and Normalisation
#--------------------------------------------------------------

SC=read.loadmat(dpath+"SC_GenCog_PROB_30.mat")
#print(SC.keys())
GrCV=SC['GrCV']   # This is used in the actual program for SC
				  # 1:34 42:75 (totally 68 regions)	

GrS=SC['GrS']     # Structural connectivity
Dist=SC['dist']   # Distance matrix?

sli=np.hstack((range(1,35),range(42,76)))
C=GrCV[sli[:,None],sli]  # look at the indexing of numpy arrays
C=C/np.amax(C)*0.2

#plt.imshow(C)
#plt.show()
print("Shape of structural connectivity matrix:", GrCV.shape)
print("Shape of SC  matrix after ROI selection is :", C.shape)

#--------------------------------------------------------------
# HCP time series data
#--------------------------------------------------------------

Data=read.loadmat(dpath+"DKatlas_timeseries.mat")
dat=Data['ts']
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
print("Shape of the filtered data is ", fdata.shape)


"""
#--------------------------------------------------------------
# Filtering of fMRI time series data
#--------------------------------------------------------------


fl =0.008           			# lowpass frequency of filter
fh =0.08            			# highpass
k=2                  			# 2nd order butterworth filter
Nq=1/(2*TR)      				# Nyquist frequency
Wn=[fl/Nq,fh/Nq] 				# Normalised frequency
b,a=butter(k,Wn,btype='band')  	# Construct the filter


Filtered_data = np.zeros((NoT,nROI,NoS))
for sub in range(NoS):
	Odata=np.squeeze(data[:,:,sub])    # BOLD of one participant
	fdata=np.zeros(Odata.shape)
	for roi in range(nROI):
		Odata[:,roi]=Odata[:,roi]-Odata[:,roi].mean()
		fdata[:,roi] =filtfilt(b,a,Odata[:,roi])
	Filtered_data[:,:,sub]=fdata
	print("Done temporal filtering for " + str(sub+1) + " participant")

np.save(dpath+"Filtered.npy",Filtered_data)
print("Filtered data stored")


plt.plot(Odata[:,27],label="Unfiltered")
plt.plot(fdata[:,27],label="Filtered")
plt.legend()
plt.show()

#plt.plot(data[:,1,27],label="Unfiltered")
#plt.plot(a[:,1,27]+data[:,1,27].mean(),label="Filtered")
#plt.legend()
#plt.show()

#--------------------------------------------------------------
# Finding sample correlation of the filtered data
#-------------------------------------------------------------- 

FCemp=np.zeros((nROI,nROI,NoS))   # Empirical functional connectivity
Pvalemp=np.zeros((nROI,nROI,NoS)) # Empirical p-values

for sub in range(NoS):
	for i in range(nROI):
		for j in range(nROI):
			samp_corr,p_val=pearsonr(fdata[:,i,sub],fdata[:,j,sub])
			FCemp[i,j,sub]=samp_corr
			Pvalemp[i,j,sub]=p_val

np.save(dpath+"FCemp.npy",FCemp)
np.save(dpath+"FCemp_pvals.npy",Pvalemp)
print("Stored the FC and Pvals")
"""
with open(dpath+"FCemp.npy",'rb') as f:
	FCemp=np.load(f)
	print("Empirical connectivity loaded properly")

#print(FCemp[:,:,1])

with open(dpath+"FCemp_pvals.npy",'rb') as f:
	FCemp_pvals=np.load(f)
	print("P-values of Empirical connectivity loaded properly")

#print(FCemp_pvals[:,:,1])


#plt.imshow(FCemp[:,:,1])
#plt.show()
#plt.imshow(FCemp_pvals[:,:,1])
#plt.show()
