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
import abagen  # For getting atalas info; this contains maany atlas info
 
participant_number=15

N_e=200
N_i=int(N_e*0.2)

dpath="/olive/Maths/R/Do/InputData/"                     # Path for structural connectivity
Ipath="/olive/Maths/R/Do/OutputData/"                    # Current path
Lconn_path='/olive/Maths/R/Do/SNN/Results/LocalConn/'    # Local connectivity path
Gloconn_path='/olive/Maths/R/Do/SNN/Results/GlobalConn/' # Global connectivity path
Weight_path='/olive/Maths/R/Do/SNN/Results/Weights/'     # Weights path
Spath='/olive/Maths/R/Do/SNN/Results/Spikes/'            # Spikes path

 
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
C=C/np.amax(C)*0.02

#plt.imshow(C)
#plt.show()
#print("Shape of structural connectivity matrix:", GrCV.shape)
print("--->Shape of SC  matrix after ROI selection is :", C.shape)

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
def NeuronParameters(R=42.5*Mohm,Vrest = -70.*mV,Vreset = -60.*mV,Vt = -50.*mV,tref = 3.*ms,
                     tauE = 15.* ms,sig=2.2*mV,dt=1*ms,pri=False):
    
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
        A[:,l] = I + sqrt(var_input)*randn(N)  
 
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

def SynapticKernel1(I_e,I_i,N_e,N_i,shift,var_input=0.0001,type="LIF"): 
    # The function SynapticKernel1 is different from SynapticKernel in that it takes both
    # excitatory and inhibitory current in to account 
    
    print("++ Forming the Synaptic Kernel..This may take a few minutes...")  
    start_scope()
    
    N_pool=N_e+N_i
    # LIF model parameters
    R, Vrest,Vreset,Vt,tref,tauE,sig,_= NeuronParameters()
    
    defaultclock.dt=1*ms
    dt=defaultclock.dt
    
    N=len(I_e)  # length of the current time series (assuming both I_e and I_i have same length
     
    A=zeros((N,N_pool))   # Time X No. of neurons
    
    for l in range(N_pool):
    	if l<N_e:
    	    A[:,l] = I_e + sqrt(var_input)*randn(N)
    	else:
    	    A[:,l] = I_i + sqrt(var_input)*randn(N) 
            
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

    final=final.astype('uint8')	  # For storage efficiency
        
    return final

def print_time(start=0,end=0,text="Elapsed time :",unit="micro"):
    
    ela =end-start
    if unit=="micro":
        ela *= 1e6
    elif unit=="milli":
        ela *= 1e3
    elif unit=="sec":
        pass
    """
    elif unit=="min":
	ela =ela/60
    elif unit=="hour":
	ela =ela/3600
    """				
    print(text,round(ela,4),unit)
 

#Parameters intialization
start_scope()

N_ROI=68

atlas=abagen.fetch_desikan_killiany()
roi = pd.read_csv(atlas['info'])
ROI_names= list(roi.iloc[np.hstack((range(1,35),range(42,76))),1])
#ROI_names= ["ROI-"+str(kk) for kk in range(N_ROI)]

defaultclock.dt=1*ms
dt=defaultclock.dt

# Load the current time series for the participants 

full_path_e=Ipath+"IExi_"+str(participant_number)+".mat"
full_path_i=Ipath+"IInh_"+str(participant_number)+".mat"

import mat73 

print("---> Participant-",str(participant_number) + " was chosen")
print("---> Loading the current time series of participant-" + str(participant_number))

It=mat73.loadmat(full_path_e)
IE=It['xn'].T
print("---> Shape of the excitatory current time series is ",IE.shape)

It=mat73.loadmat(full_path_i)
II=It['xg'].T
print("---> Shape of the inhibitory current time series is ",II.shape)

sig_len= 60000  # in msec

shift=5           # Time series shift for avoiding initial spikes
N= sig_len + shift  # Total number of time points in the input current
IE=IE[:N,:]
II=II[:N,:]


(N_e,N_i,p_local,p_global,N_pool,M_local,M_global,f_e,f_i,f_ei,f_ie) = pool_parameters(N_e,N_i,0.1,0.1)
pool_summary(N_e,N_i,p_local,p_global) 

#neuron_summary()

print("---> Length of current time series" ,N, "with", dt, "sampling time \n")
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

from scipy.signal import savgol_filter 

Ivar=1e-4
for l in range(N_ROI):  #N_ROI
	IE[:,l]=savgol_filter(IE[:,l],101,3)  # Smoothening with the filter
	II[:,l]=savgol_filter(II[:,l],101,3)  # Smoothening with the filter
	Curr[:,:,l],_,A[:,:,l]= SynapticKernel1(IE[:,l],II[:,l],N_e,N_i,shift,var_input=Ivar,type="LIF")  # Curr is for one whole ROI output and A is input 
	True_rate[:,l]=IFcurve(aE,bE,dE,IE[:,l]*namp) # True rate (from IF curve)
	print("++ Done with ROI" + str(l+1))
	line_print(50)    

#-------------------------------------------------------------- 
# Establishing Synaptic connectivity 
#  Local pool:  ROI-wise synaptic connectivity matrix
#-------------------------------------------------------------- 
print("---> Establishing synaptic connectivity for local pool ")
start=time.time()
Local_con=np.zeros((N_pool,N_pool,N_ROI)).astype('uint8')   # For storage efficiency
for l in range(N_ROI):
    Local_con[:,:,l]=Synpatic_Connectivity(N_e,N_i,p_local,region="local")   
end=time.time() 
print_time(start,end,text="---> Time taken for "+str(N_ROI)+ " ROIs is : ",unit="sec") 

print("---> Saving local synpatic connectivity")
np.save(Lconn_path+'Participant'+str(participant_number)+'.npy',Local_con)

#-------------------------------------------------------------- 
# Long-range: Global connecitivity to each ROI
#-------------------------------------------------------------- 

print("---> Establishing synaptic connectivity for global pool")
start=time.time()
Lrange_con=np.zeros((N_pool,N_e,N_ROI-1,N_ROI)).astype('uint8') 
for l in range(N_ROI):
    for m in range(N_ROI-1):
        Lrange_con[:,:,m,l]=Synpatic_Connectivity(N_e,N_i,p_global,region="global")       
end=time.time() 
print_time(start,end,text="---> Time taken for "+str(N_ROI)+ " ROIs is : ",unit="sec")
 

print("---> Establishing synaptic connectivity for local pool ")
np.save(Gloconn_path+'Participant'+str(participant_number)+'.npy',Lrange_con)

#-------------------------------------------------------------- 
# Formation of Synaptic Kernel Matrix
#-------------------------------------------------------------- 

def get_SK(Ind,ROI_Ind,conn_type="local"):          # Includes the negative inhibitory currents
    
    if conn_type=="local": 
        Synaptic_Index=Local_con[Ind,:,ROI_Ind].nonzero()[0]  # Synaptic connection (Excitatory + Inhibitory)
        Synaptic_Kernel = Curr[:,Synaptic_Index,ROI_Ind]   # Get the corresponding synaptic current
        #print(Synaptic_Kernel.shape)   

        # Index of excitatory and Inhibitory synapese
        Lin=len(where(Synaptic_Index>=N_e)[0]) # Inhibitory Index
        Lex=len(where(Synaptic_Index<N_e)[0])  # Excitatoru Index
        return (Synaptic_Kernel,Lex,Lin,Synaptic_Index)
    
    elif conn_type=="global":
        Synaptic_Kernel=[]
        Inhi,Exci=[],[]
        Other=list(range(N_ROI))
        Other.remove(ROI_Ind) 
        
        for l in range(N_ROI-1):            
            Synaptic_Index=Lrange_con[:,Ind,l,ROI_Ind].nonzero()[0] # Synaptic connection (Excitatory + Inhibitory)
            Synaptic_Kernel.append(Curr[:,Synaptic_Index,l])   # Get the corresponding synaptic current
                
            Inhi.append(len(where(Synaptic_Index>=N_e)[0])) # Inhibitory Index
            Exci.append(len(where(Synaptic_Index<N_e)[0]))  # Excitatoru Index
            
        return (Synaptic_Kernel,Exci,Inhi) 
    else:
        print("ERROR: **Connection type must be either global or local**") 

 


#--------------------------------------------------------------	
# Weight calculation procedure here
# Here we compute the weights  
#--------------------------------------------------------------	
import random
start=time.time() 

# Final weights as a list
Wroi=[None]*N_ROI*4  			# 4 different weights 
W_mtx=np.zeros((N_pool,N_pool,N_ROI))   # Final weight matrix

for roi in range(N_ROI):    # For each ROI 
    Iext= A[shift:,:,roi]
    I_new =Iext 
    
    WEE,WIE,WEI,WII=[],[],[],[] 

    for Ind in range(N_pool):           # For each neuron in the pool  

        # Current time series to that neuron 

        # Local pool synpatic kernel
        Synaptic_Kernel,Lex,Lin,Synaptic_Index = get_SK(Ind,roi,"local")  # N X N_pool X N_ROI np.array (Ind is source index)

        # Long-range connectivity considerations 
        # for Excitatory neurons?) 
        if (Ind< N_e):  
            Longrange_Kernel,Exci,Inhi = get_SK(Ind,roi,"global") # (N_ROI-1) List of N X N_pool np.array (Ind is source index)

            ## Get the list of Index of ROIs other than ROI_Ind
            Other=list(range(N_ROI))
            Other.remove(roi) 

            for l in range(N_ROI-1): 
                LRex =Longrange_Kernel[l][:,:Exci[l]] 
                LRin = Longrange_Kernel[l][:,Exci[l]:] 

                I_new[:,Ind]= I_new[:,Ind]- (muEE * dot(LRex,np.repeat(C[roi,Other[l]],Exci[l])))  # Zeros here is the ROI-0 local population
                I_new[:,Ind]= I_new[:,Ind]- (muEI * dot(LRin,np.repeat(C[roi,Other[l]],Inhi[l]))) 

            ## Find synaptic weights using non-negative least squares

            W=find_w(Synaptic_Kernel,I_new[:,Ind])  # After removing the contributions from long range 
            WEE.append(W[:Lex])
            WEI.append(W[Lex:])   # Negative Weights here (negated during synapse)
            W_mtx[Ind,Synaptic_Index,roi]=np.concatenate((W[:Lex],-W[Lex:]))
            
        # For Inhibitory  neurons    
        else: 
            W=find_w(Synaptic_Kernel,I_new[:,Ind])
            WIE.append(W[:Lex])
            WII.append(W[Lex:])  # Negative Weights here(negated during synapse formation in the network)
            W_mtx[Ind,Synaptic_Index,roi]=np.concatenate((W[:Lex],-W[Lex:]))
 

        print("---> Computed weights for Neuron-{} in ROI-{}".format(Ind,roi))
        
        # Restoring the current (this is needed, because we get positive weights in the optimization)
    Curr[:,N_e:,roi]=-Curr[:,N_e:,roi]
         
    N_WEE=len([item for sublist in WEE for item in list(sublist)]) 
    N_WEI=len([item for sublist in WEI for item in list(sublist)])
    N_WIE=len([item for sublist in WIE for item in list(sublist)])
    N_WII=len([item for sublist in WII for item in list(sublist)])
        
    print("EE :",N_WEE)
    print("EI :",N_WEI)
    print("IE :",N_WIE)
    print("II :",N_WII)
    
    
    temp_Ind=roi + (roi*(3))
    Wroi[temp_Ind]=WEE
    Wroi[temp_Ind+1]=WII
    Wroi[temp_Ind+2]=WIE
    Wroi[temp_Ind+3]=WEI  


print("---> Done with synaptic weight calculation \n")
end=time.time() 
print_time(start,end,text="---> Time taken for  synaptic weight caluation for "+str(N_ROI)+ " ROIs and "+ str(N_pool)+" neurons in each ROI is : ",unit="sec")


print("---> Saving the synaptic weights ...\n")
np.save(Weight_path+'Participant'+str(participant_number)+'.npy',W_mtx)



print("---> Proceeding with Network Simulation ")
start=time.time() 
#--------------------------------------------------------------	
# Staring of SNN simulation
#--------------------------------------------------------------	

#NeuronParameters()

start_scope()
defaultclock.dt  = 1*ms
dt=defaultclock.dt

run_time=N
dlocal=0*ms 
 
(R, Vrest,Vreset,Vt,tref,tauE,sig,dt)= NeuronParameters()
 

## Forming the input matrix for each ROI
I_exci=np.zeros((N,N_e*N_ROI)) 
I_inhi=np.zeros((N,N_i*N_ROI)) 

for l in range(N_ROI):
    I_exci[:,l*N_e:((l+1)*N_e)]=A[:,:N_e,l] 
    I_inhi[:,l*N_i:((l+1)*N_i)]=-A[:,N_e:,l]   # Input current to the inhibitory population is negated
 
I_exci = TimedArray(I_exci*nA,dt=dt)
I_inhi = TimedArray(I_inhi*nA,dt=dt)

    
# Excitatory model
EqE=Equations('dV/dt=(-(V-Vrest) + (I_exci(t,i)*R) )*(1./tauE)  + (sig*(1./tauE)**0.5)*xi : volt (unless refractory)') 

# Inhibitory model
EqI=Equations('dV/dt=(-(V-Vrest) + (I_inhi(t,i)*R) )*(1./tauE)  + (sig*(1./tauE)**0.5)*xi : volt (unless refractory)')  

 ## Neuron group for all ROI

E = NeuronGroup(N_e*N_ROI,                  # Number of excitatory (for all ROI)
                method='euler', 
                model=EqE, 
                threshold='V > Vt', 
                reset='V=Vreset', 
                refractory='tref')

I = NeuronGroup(N_i*N_ROI,                  # Number of  inhibitory (for all ROI)
                method='euler', 
                model=EqI, 
                threshold='V > Vt', 
                reset='V=Vreset', 
                refractory='tref')  
 
#--------------------------------------------------------------    
# Assigning neruons to each ROI (Neuron subgroup type)
#-------------------------------------------------------------- 

Exc = [ E[y*N_e:((y+1)*N_e)] for y in range(N_ROI)]
Inh = [ I[z*N_i:(z+1)*N_i] for z in range(N_ROI)]  

# Initilizing parameters

EE_loc, II_loc, EI_loc, IE_loc = [None]*N_ROI, [None]*N_ROI, [None]*N_ROI, [None]*N_ROI 
EI_lr, EE_lr =[], []

print("---> Establishing synaptic connections")

for h in range(N_ROI): 
    
    ## LOCAL CONNECTION IN ALL THE POOLS ## 
    # Recurrent excitation and inhibition
    EE_loc[h] = Synapses(Exc[h], Exc[h], 'w:1', delay = dlocal, on_pre='V_post=V_post +(V_pre*w)')  
    targets,sources=Local_con[:N_e,:N_e,h].nonzero() 
    EE_loc[h].connect(i=sources,j=targets)
    WEE=[item for sublist in Wroi[h + (h*(3))] for item in list(sublist)]
    EE_loc[h].w = WEE

    II_loc[h] = Synapses(Inh[h], Inh[h], 'w:1', delay = dlocal, on_pre='V_post=V_post +(V_pre*w)') 
    targets,sources=Local_con[N_e:,N_e:,h].nonzero() 
    II_loc[h].connect(i=sources,j=targets)
    WII = [-item for sublist in Wroi[h + (h*(3))+1] for item in list(sublist)]  # FIC
    II_loc[h].w = WII    

    # feeback excitation and inhibition
    EI_loc[h] = Synapses(Exc[h], Inh[h], 'w:1', delay = dlocal, on_pre='V_post=V_post +(V_pre*w)')  
    targets,sources=Local_con[N_e:,:N_e,h].nonzero() 
    EI_loc[h].connect(i=sources,j=targets) 
    WIE = [item for sublist in Wroi[h + (h*(3))+2] for item in list(sublist)]
    EI_loc[h].w = WIE

    IE_loc[h] = Synapses(Inh[h], Exc[h], 'w:1', delay = dlocal, on_pre='V_post=V_post +(V_pre*w)')    
    targets,sources=Local_con[:N_e,N_e:,h].nonzero() 
    IE_loc[h].connect(i=sources,j=targets) 
    WEI = [-item for sublist in Wroi[h + (h*(3))+3] for item in list(sublist)]   # FIC
    IE_loc[h].w = WEI  

    # For each local pool there is a long range connection

    for m in range(N_ROI-1):
     
          EE_lr, EI_lr= None, None

          EE_lr = Synapses(Exc[m],Exc[h], 'w:1', on_pre='V_post=V_post +(V_pre*w)') 
          sources,targets=Lrange_con[:N_e,:N_e,m,h].nonzero() 
          EE_lr.connect(i=sources,j=targets)
          EE_lr.w=  muEE * C[h,m]

          EI_lr = Synapses(Inh[m],Exc[h], 'w:1', on_pre='V_post=V_post +(V_pre*w)') 
          sources,targets=Lrange_con[N_e:,:N_e,m,h].nonzero()
          EI_lr.connect(i=sources,j=targets)  
          EI_lr.w=  muEI * C[h,m]
            
          ## Delays between the long range connections  
            
          #exc_lr_itoj.w =  (1 + eta * hier[m]) * muEE * fln_mat[m,h]
          #etoi_lr_itoj.w = (1 + eta * hier[m]) * muIE * fln_mat[m,h]

          # Delays between ROIs      
          #meanlr, varlr = delayMat[m,h], .1*delayMat[m,h]
          #EE_lr.delay = np.random.normal(meanlr,varlr,len(exc_lr_itoj.w))*ms
          #EI_lr.delay = np.random.normal(meanlr,varlr,len(etoi_lr_itoj.w))*ms 

print("---> Done with establishing connection \n")
print("---> Running the network \n")

#M=StateMonitor(E,'V',record=True) 
S=SpikeMonitor(E)
run(N*ms)

end=time.time() 
print_time(start,end,text="---> Time taken for running the network with  "+str(N_ROI)+ " ROIs and "+ str(N_pool)+" neurons in each ROI is : ",unit="sec")

print("---> Calculating spikes from the computed network \n")

# Variable S contains the spike information

netspike = len(S.i)
allspike = np.empty([netspike,2])
allspike[:,0]=S.t/ms               # events
allspike[:,1]=S.i                # Neuron number
allspikesorted = allspike[allspike[:,1].argsort(),]

allspikesorted  # 2D array of spike times on 1st column and 
                # neuron number on the 2nd column

# The variable 'allspikesorted' is exploited here to arrange it in the list of area spikes
# So Spk is the Spike information in the list with each area is an element of the list
# An area contains Nn number of neurons

Spk=[]  # This is list of 2D array. Each 2D array is for an area that contains spikes

for k in range(N_ROI):  
    if not k:
        Ind1=np.where(allspikesorted[:,1]<(N_e-1)*(k+1)+k)[-1][-1]
        Spk.append(allspikesorted[0:Ind1,:])
    else:
        Ind2=np.where(allspikesorted[:,1]<(N_e-1)*(k+1)+k)[-1][-1] 
        Spk.append(allspikesorted[(Ind1+1):Ind2,:])
        Ind1=Ind2 

print("---> Computing 0 and 1 spike train \n") 

def SpkTime2Binary(inp,N):
    res=np.zeros(N)
    res[inp]=1
    res.astype('ubyte')
    return res

SPK_TRAINS=np.zeros((sig_len,N_e,N_ROI),dtype='ubyte') 

for roi in range(N_ROI):
	for neu in range(N_e):
		spk_train= Spk[roi][Spk[roi][:,1]==neu+(roi*N_e),0].astype('uint16')
		spk_train=spk_train[spk_train<(sig_len-1)] # To make sure no spikes after sig_len
		SPK_TRAINS[:,neu,roi]=SpkTime2Binary(spk_train,sig_len)
	print("---> Storing 0 and 1 spikes of ROI-"+str(roi))

print("---> Saving the spike time series...")
np.save(Spath+'SPK_TRAIN_Participant'+str(participant_number)+'.npy',SPK_TRAINS)

"""
maxrate = np.empty([N_ROI,1])     # max rate
meanrate = np.empty([N_ROI,1])    # mean rate

binsize =10*ms   
duration=N*ms

netbinno = int(1+(duration/ms)-(binsize/ms))
poprate = np.empty([N_ROI,netbinno ])   #pop-rate (instantaneous rate)

count = 0               #for each spike. 
stepsize = 1*ms
monareaktimeall = []
for u in range(N_ROI):  # For each ROI
      monareaktime = []

      while((count < netspike) and (allspikesorted[count,1]<(N_e)*(u+1)) ):
        monareaktime.append(allspikesorted[count,0])#append spike times. for each area.
        count = count + 1

      vals= []
      vals = numpy.histogram(monareaktime, bins=int(duration/stepsize))    
      valszero = vals[0]  #now valsbad[0] is a big vector of 500 points. binsize/stepsize = aplus say.
      astep = binsize/(1*ms)
      valsnew = np.zeros(netbinno)    
      acount = 0
        
      while acount < netbinno:        
          valsnew[acount] = sum(valszero[acount:int(acount+astep)])
          acount=acount+1

      valsrate = valsnew*((1000*ms/binsize) /(N_e) ) #new divide by no of neurons per E pop. 
      poprate[u,:] = valsrate   
      maxrate[u,0] = max(valsrate[int(len(valsrate)/3):])
      monareaktimeall.append(monareaktime)
    
print("++ Plotting the rates \n")
Nroi=5


True_another=np.zeros((N,Nroi))

for l in range(Nroi):
    True_another[:,l]=IFcurve(aE,bE,dE,IE[:,l]*namp)

area_name_list  = ROI_names[:Nroi]
area_idx_list   = [ROI_names.index(name) for name in area_name_list]
f, ax_list      = plt.subplots(len(area_idx_list), sharex=True)
 
for ax, area_idx in zip(ax_list, area_idx_list):
  
    y_plot  = poprate[area_idx,:] 
    txt     = ROI_names[area_idx]

    y_plot = y_plot - y_plot.min()
    ax.plot(y_plot,"r",label="Spike Model")
    #ax.plot(True_rate[:,area_idx],"b",label="Rate-model")
    ax.plot(True_another[:,area_idx],"b",label="Rate-model")
    ax.text(0.8, 0.8, txt, transform=ax.transAxes,size=16)

    
    #ax.set_yticks([y_plot.max()])
    #ax.set_yticklabels(['{:0.1f}'.format(y_plot.max())])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim(1,sig_len)
    ax.set_ylim(0,10)
    if area_idx==0:	
        ax.legend()

f.text(0.09, 0.5, 'Firing rate (Hz)',  rotation='vertical',size=16)
f.set_size_inches(18.5, 10.5)
ax.set_xlabel('Time (ms)',size=16)
show()

area_name_list  =  ROI_names[:Nroi]
area_idx_list   = [ROI_names.index(name) for name in area_name_list]
f, ax_list      = plt.subplots(len(area_idx_list), sharex=True)
 
for ax, area_idx in zip(ax_list, area_idx_list):
 
    y_plot  = Spk[area_idx][:,0]
    txt     = ROI_names[area_idx]

    y_plot = y_plot - y_plot.min()
    ax.plot(Spk[area_idx][:,0],Spk[area_idx][:,1]-((area_idx)*N_e)+5,'r.',markersize=1)
    ax.text(0.95, 0.95, txt, transform=ax.transAxes,size=14)

    
    #ax.set_yticks([0,200,400])
    #ax.set_yticklabels(['{:0.1f}'.format(y_plot.max())])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim(1,sig_len)
    #ax.legend()

f.text(0.09, 0.5, 'Neuron Index', va='center', rotation='vertical',size=16)
f.set_size_inches(18.5, 10.5)
ax.set_xlabel('Time (ms)',size=16)
show()


#exec(open("SNN_try2.py").read())
"""

