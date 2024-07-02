# Import the necessary packages
from brian2 import * 
import numpy as np
import pandas as pd
import random
import time

import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter,filtfilt  # For temporal filtering
from scipy.stats import pearsonr				  # For calulating r, and pvalue
import scipy.io as read   # for reading matlab files
import h5py
import mat73 as mat

dpath="/olive/Maths/R/Do/InputData/"  # Path for structural connectivity

NoS=389	   # Number of participants	
TR=0.754   # [s] HCP data TR
NoT=616   # Length of time series data

#--------------------------------------------------------------
# Loading Structural connectivity data and Normalisation
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
#print("Shape of structural connectivity matrix:", GrCV.shape)
print("Shape of SC  matrix after ROI selection is :", C.shape)

# import the user-defined functions for this project
# simple spike counter function
def SC(S,n):
    if all(S.t == [0.]*second):
        count=0
    else:
        count=len(S.t)-n
    return count 


## Specify an IF curve (we adopted the model from Dermitas et al)

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

# Example of IF curve
#I=arange(0,0.3,0.01)*namp   # This is the current range for which there is a transfer
#r=IFcurve(a,b,d,I)  # This is the desired rate 


def ItoR(I,ntype="LIF"):
    if ntype=="LIF": 
        return np.array([LIF_ItoR(l) for l in I])

    
def LIF_ItoR(I):
    start_scope() 
    N=1  
   
    run_time=200*ms
    
    R, Vrest,Vreset,Vt,tref,tau,_= NeuronParameters()

    eqs = '''dV/dt = ((Vrest - V) + (I*R))/tau : volt (unless refractory)
             I : amp'''

    group = NeuronGroup(N, eqs,
                        threshold='V > -50*mV',
                        reset='V =-70*mV',
                       refractory='tref',
                       method="euler")


    S= SpikeMonitor(group) 
    group.I=I
    run(run_time)
    r=SC(S,N)/(run_time*N)   # rate
    return r 
    
    
def weight_hist(W):
    fig,ax=subplots(1,3,figsize=(10,3))
    ax[0].hist(W[:f_e],bins=30)
    ax[0].title.set_text("Excitatory weights") 
    ax[1].hist(W[f_e:(f_e+f_i)],bins=30)
    ax[1].title.set_text("Inhibitory weights") 
    ax[2].hist(W,bins=30)
    ax[2].title.set_text("Total Weights") 
    show()


def plot_all(Iext_list,out_spike,S,M): 
    figure()
    plot(Iext_list/nA,out_spike,'r*-')
    xlabel("Itotal (nA) ")
    ylabel("Rate (Hz)")
    grid()
    show()


    figure()
    plot(M.t/ms,M.V[0],'k')
    xlabel('Time (ms)')
    ylabel('Voltage'); 


    figure()
    plot(S.t/ms, S.i,'k.')
    xlabel('Time (ms)')
    ylabel('Neuron index');
    
def rate_from_spikes(S,bin_size=1*ms):
    No=len(unique(S.i))   # Number of neurons    

    event=S.t
    Ma=max(S.t)
    bin_size=1*ms

    start=1*ms
    rate=[]

    while (start < Ma-(5*bin_size)):
        aa=(event>= start )
        bb=(event <= start + bin_size) 
        rate.append(sum((aa & bb)*1)/(bin_size*No))
        start =start + 1*ms

    return rate    


def Cmtx(N_ROI): 
    C=np.zeros((N_ROI,N_ROI))
    for l in range(N_ROI):
        C[l,:]=roll(abs((1-np.exp(-(arange(N_ROI)*randn()/0.25)))*0.001),l)
    return(C)


# Default parameters of a LIF neuron
def NeuronParameters(R=63*Mohm,Vrest = -70.*mV,Vreset = -60.*mV,Vt = -50.*mV,tref = 3.*ms,
                     tauE = 15.* ms,sig=3.7*mV,dt=1*ms,pri=False):
    
    if(pri):
        print("Neuron Parameters (R, Vrest,Vreset,Vt,tref,tauE,sig,dt) are  : ",(R, Vrest,Vreset,Vt,tref,tauE,sig,dt))
    
    return (R,Vrest, Vreset,Vt,tref,tauE,sig,dt)
    
#(R, Vrest,Vreset,Vt,tref,tauE,dt)=NeuronParameters()    

#--------------------------------------------------------------
# Synpatic Kernel construction
# Each column is a time series of membrane current with no other synaptic input
# Input current I should be in nA
#-------------------------------------------------------------- 

def SynapticKernel(I,N_pool,shift,var_input=0.0001,type="LIF"): 
    
    print("++ Forming the Synaptic Kernel..This may take a few minutes...")  
    start_scope()
    # LIF model parameters
    R, Vrest,Vreset,Vt,tref,tauE,sig,_= NeuronParameters()
    
    defaultclock.dt=1*ms
    dt=defaultclock.dt
    
    N=len(I)  # length of the current time series
     
    A=zeros((N,N_pool))   # Time X No. of neurons
    for l in range(N_pool): 
        A[:,l] = I #+ sqrt(var_input)*randn(N)  
 
    # Input current array to neurons
    B = TimedArray(A*nA, dt=defaultclock.dt)  
    
    eqsE = Equations('''
      dV/dt=(-(V-Vrest) + (B(t,i)*R) )*(1./tauE)  + (sig*(1./tauE)**0.5)*xi : volt (unless refractory)
      ''' ) 

    E = NeuronGroup(N_pool,     
                #method='euler', 
                model=eqsE, 
                threshold='V > Vt', 
                reset='V=Vreset', 
                refractory='tref')

    M=StateMonitor(E,'V',record=True)   
     
    run(N*ms)  
    
    MemPotential=M.V.T[shift:]  # membrane Voltage
	
    MemCurrent=(MemPotential-Vrest)/R   # membrane current
    
    print("++ Shape of the Synaptic Kernel is X ", MemCurrent.shape)  
     
    return (MemCurrent,MemPotential,A) 
    #return MemCurrent

def line_print(n=100):
    print("-"*n)
 
def check_reconstruction(A,x,y,xmin=0,xmax=1000):
    Recons=np.dot(A,x) # Reconstruciton for checking purpose
    figure(figsize=(20,4))
    plot(y*nA,'r',label="Original")
    plot(Recons,'b-',label="Reconstruced")
    xlim(xmin,xmax)
    legend()
    show() 
    
def find_w(A,x):
    from scipy.optimize import nnls

    W=nnls(A/nA,x.ravel())
    return W[0]    
    
#def makeW(W,Index):
    # Function to keep the postive weights and make the negative weights
#    W[f_e:(f_e+f_i)]=-W[f_e:(f_e+f_i)] 
#    return W

def pool_parameters(N_e=100,N_i=20,p_local=0.1,p_global=0.1):
    f_e= int(p_local*N_e*N_e) # Fraction of recurrent excitatory neurons
    f_i= int(p_local*N_i*N_i) # Fraction of recurrent inhibitory neurons
    f_ei=int(p_local*N_i*N_e)
    f_ie=int(p_local*N_e*N_i)
    M_global= int(p_global * ((N_e*N_e)+(N_e*N_i)))* (N_ROI-1)  # Total number of long range connections (except its own region)
    M_local = f_e + f_i + f_ei + f_ie # Total number of local synapse in a pool
    return (N_e,N_i,p_local,p_global,N_e+N_i,M_local,M_global,f_e,f_i,f_ei,f_ie)

def pool_summary(N_e=1600,N_i=400,p_local=0.1,p_global=0.1):
    (N_e,N_i,p_local,p_global,N_pool,M_local,M_global,f_e,f_i,f_ei,f_ie) = pool_parameters(N_e,N_i,p_local,p_global)

    line_print()
    print("++ Number of neurons in a pool ",N_pool)
    line_print()
    print("Local synapse summary: ")
    line_print(27)
    print("For local connection probability of ",p_local)
    print("Total number of local synapse in a pool is",M_local) 

    print("Total number of  EE is {} and  II {} ".format(f_e,f_i))
    print("Total number of  EI is {} and  IE {}".format(f_ei,f_ie))
    print("\n")
    print("Long-range synapse summary: ")
    line_print(28)
    print("For ", N_ROI, " pools")
    print("with the long-range connection probability of ",p_global)
    print("Number of long-range synapses to a pool is",M_global) 
    line_print() 

from numpy.random import *  

def Synpatic_Connectivity(N_e,N_i,p,region="local"):
    
    # Generating N_pool X N_pool matrix
    if region=="local": 
        mat1=choice([0, 1], size=(N_e,N_e), p=[1-p,p])
        mat2 =choice([0, 1], size=(N_e,N_i), p=[1-p,p])
        mat3=choice([0, 1], size=(N_i,N_e), p=[1-p,p])
        mat4= choice([0, 1], size=(N_i,N_i), p=[1-p,p]) 

        mat5=np.concatenate((mat1,mat2),axis=1)
        mat6=np.concatenate((mat3,mat4),axis=1)
        final=np.concatenate((mat5,mat6),axis=0)
        
    # Generating N_pool X N_e  Global matrix    
    else:
        mat1=choice([0, 1], size=(N_e,N_e), p=[1-p,p])
        mat3 =choice([0, 1], size=(N_i,N_e), p=[1-p,p])
        final=np.concatenate((mat1,mat3),axis=0)
        
    return final

def print_time(start=0,end=0,text="Elapsed time :",unit="micro"):
    
    ela =end-start
    if (unit=="micro"):
        ela *= 1e6
    elif (unit =="milli"):
        ela *= 1e3
    elif (unit=="sec"):
        pass
    print(text,round(ela,3),unit)
 

#Parameters intialization
start_scope()

N_ROI=68

ROI_names= ["ROI-"+str(kk) for kk in range(N_ROI)]

defaultclock.dt=1*ms
dt=defaultclock.dt

# Load the current time series for the participants

Ipath="/olive/Maths/R/Do/OutputData/"
participant_number=1

full_path_e=Ipath+"IExi_"+str(participant_number)+".mat"
full_path_i=Ipath+"IInh_"+str(participant_number)+".mat"


import mat73 

print("++ Participant-",str(participant_number) + " was chosen")
print("++ Loading the current time series of participant-" + str(participant_number))

It=mat73.loadmat(full_path_e)
IE=It['xn'].T

print("Shape of the current time series is ",IE.shape)
#It=mat73.loadmat(full_path_i)
#II=It['xg'].T
#II=II[:5000,:]


#print(II.shape)

sig_len= 5000  # in msec

shift=5           # Time series shift for avoiding initial spikes
N= sig_len + shift  # Total number of time points in the input current
IE=IE[:N,:]


N_e=200
N_i=int(N_e*0.2)
(N_e,N_i,p_local,p_global,N_pool,M_local,M_global,f_e,f_i,f_ei,f_ie) = pool_parameters(N_e,N_i,0.1,0.1)
pool_summary(N_e,N_i,p_local,p_global) 

#neuron_summary()

print("++ Length of current time series" ,N, "with", dt, "sampling time \n")
line_print() 

# Global parameters
muEE,muEI = 0.1,0.5 # Global scaling parameter 


Curr=np.zeros((sig_len,N_pool,N_ROI))  # N X N_pool X N_ROI

#AE=np.zeros((N,N_e,N_ROI))
#AI=np.zeros((N,N_i,N_ROI))
True_rate = np.zeros((N,N_ROI))
A=zeros((N,N_pool,N_ROI))


(aE,bE,dE)=IFpara(name="Deco_E")
#fg,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
#ax1.plot(IE[:10000,0])
#ax2.plot(II[:10000,0])
#(aE,bE,dE)=IFpara(name="Deco_E")
#ax3.plot(IFcurve(aE,bE,dE,IE[:10000,0]*namp))
#(aI,bI,dI)=IFpara(name="Deco_I")
#ax4.plot(IFcurve(aI,bI,dI,II[:10000,0]*namp))
#plt.show()

Ivar=1*e-4
for l in range(N_ROI):  #N_ROI
	Curr[:,:,l],_,A[:,:,l]= SynapticKernel(IE[:,l],N_pool,shift,var_input=Ivar,type="LIF")  # Curr is for one whole ROI output and A is input
	Curr[:,N_e:,l]=-Curr[:,N_e:,l]  # For ihhibitory
	#Curr[:,N_e:,l],_,AI[:,:,l]=SynapticKernel(II[:,l],N_i,shift,var_input=Ivar,type="LIF") 
	True_rate[:,l]=IFcurve(aE,bE,dE,IE[:,l]*namp)                                     # True rate (from IF curve)
	print(" ++ Done with ROI" + str(l+1))
	line_print() 

 

