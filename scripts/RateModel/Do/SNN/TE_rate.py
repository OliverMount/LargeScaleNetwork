#--------------------------------------------------------------	
# Program to compute and store the transfer entropy for rate time series
#-------------------------------------------------------------- 

import matplotlib.pyplot as plt
import numpy as np
import os
import re

def min_max(T):
	mi=np.amin(T)
	ma=np.amax(T)
	return (T-mi)/(ma-mi)

from pyinform import transfer_entropy as te
import pandas as pd
import seaborn as sns

N_ROI=68
NoS=389

Rpath='/olive/Maths/R/Do/OutputRate/'
Fpath='/olive/Maths/R/Do/'

Files=os.listdir(Rpath)  # Files in the rate path
plist=[]
for k in Files:
    plist.append(int(re.findall('\d+',k)[0]))

Lp=len(plist)
print("Total number of participants considered is " , len(plist))

#history_points=list(range(1,4))
history_points=[1]
L=len(history_points)

slist=[5,20,50,100,150,200,250,300,350,400,500,720]
Ls=len(slist)

TE=np.zeros((N_ROI,N_ROI,Ls,L,Lp))
#plist=list(range(1,NoS+1))

kp=0
for participant_number in plist:
	
	print('++ Loading participant-'+str(participant_number)+' Rate data')
 
	with open(Rpath+'Participant'+str(participant_number)+'.npy', 'rb') as f: 
		Rate=np.load(f)

	ks=0
	for skip_point in slist:
		for source in range(N_ROI):
			#print("++++ Determining TE of Source ROI-",source)
			for target in range(N_ROI):
				#print("++++++ Source: " ,source,"  Target: ",target)
				kh=0
				for k in history_points:
					TE[source,target,ks,kh,kp]=te(Rate[500::skip_point,source],Rate[500::skip_point,target],k)
					kh=kh+1	
		ks=ks+1  
		print('++ Computed TE for time resoultion of -'+str(skip_point)+' msec')
	print('++ Computed TE for participant-'+str(participant_number))
	kp=kp+1 
		#Avg_TE=min_max(np.mean(TE,axis=2))
		#print(TE[20,0,0],TE[0,20,0])
		#print(Avg_TE[20,0],Avg_TE[0,20])
		#plt.imshow(Avg_TE,cmap="jet");plt.colorbar();plt.show()
		#fname='TE_'+str(participant_number)+'_'+str(skip_point)
		#plt.imsave(Fpath+fname+'.png',Avg_TE)


print("++ Saving the Transfer entropy file")
np.save("TE.npy",TE)
