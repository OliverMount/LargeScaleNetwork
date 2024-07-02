#--------------------------------------------------------------
# Program to obtain rate time series from current time series
# and stores them in a dedicated path
#--------------------------------------------------------------


# Import the necessary packages
from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mat73


Ipath="/olive/Maths/R/Do/OutputData/"
Rpath="/olive/Maths/R/Do/OutputRate/"

Timepoints=240000    # 4 mins time data
N_ROI=68
NoS=389	   # Number of participants	
TR=0.754   # [s] HCP data TR
NoT=616   # Length of time series data

#--------------------------------------------------------------	
# necessary functions
#-------------------------------------------------------------- 

def IFpara(name="Derm_E"):
	if name=="Derm_E":
		a=310/ncoulomb
		b=125*Hz
		d=0.16*second
	elif name=="Derm_I":
		a=615/ncoulomb
		b=177*Hz
		d=0.087*second
	elif name=="Deco_E":
		a=310/ncoulomb
		b=125*Hz
		d=0.16*second
	elif name=="Deco_I":
		a=615/ncoulomb
		b=177*Hz
		d=0.087*second
	else:
		print("No parameter set available for the type:" + name + ". Please check your name type.")
	return (a,b,d)	

def IFcurve(a,b,d,I):
    return [((a*l) -b)/(1-exp(-d*((a*l)-b))) for l in I]


# Load the current time series for the participants
for participant_number in range(1,NoS):
	print("---> Participant-",str(participant_number) + " was chosen")
	print("---> Loading the current time series of participant-" + str(participant_number))

	full_path_e=Ipath+"IExi_"+str(participant_number)+".mat"
	
	It=mat73.loadmat(full_path_e)
	IE=It['xn'].T

	I=IE[:Timepoints,: ]
	
	del IE
	del It
	print("---> Shape of the current time series is ",I.shape)

	Rate=np.zeros(I.shape)
	(aE,bE,dE)=IFpara(name="Deco_E")

	print("---> Computing the rate time series for this participant ...")
	
	for l in range(N_ROI):
		Rate[:,l]=IFcurve(aE,bE,dE,I[:,l]*namp)
		print("---> Done with ROI- ...",l)

	print("---> Saving the  rate time series...")
	numpy.save(Rpath+"Participant"+str(participant_number)+".npy",Rate)
