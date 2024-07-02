#--------------------------------------------------------------
# Loading necessary modules
#--------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.optimize import least_squares  # For solving the steady state problem
import h5py
import matplotlib.pyplot as plt

#--------------------------------------------------------------
# Step1. Load the data 
#--------------------------------------------------------------

ipdir = "/olive/Maths/R/De/data/"
opdir ="/olive/Maths/R/De/output/" 

fname= ipdir+"demirtas_neuron_2019.hdf5"

with h5py.File(fname,"r") as f:
    # List all groups
    KEYS=f.keys()	
    #print("Keys: %s" % KEYS)
    SC = f['sc'][()]             # 1a. Load Strctural connectivity
    T1WT2W = f['t1wt2w'][()]     # 1b. Load  T1w/T2W  

# Normalise the SC here

def SC_normalize(SC):
	SCnormlized=np.zeros(SC.shape)
	ma=np.amax(SC)
	mi=np.amin(SC)
	for l in range(SC.shape[0]):
		SCnormlized[l]=(SC[l]-mi)/(ma-mi)	
	return SCnormlized
	
SC=SC_normalize(SC)
print("Maximum of SC",np.amax(SC))
print("Minimum of SC",np.amin(SC))

#-------------------------------------------------------------- 
# Step2a. Set up the rate-model parameters
#-------------------------------------------------------------- 

nROI=SC.shape[0]
ROInames=["ROI-"+str(k+1) for k in range(nROI)]

# fixed model weights here

wE=1   # Weights for the background current
wI=0.7
Ib = 0.382        # nA

wEE=0.08 
wEI=0.05
wII=0.04
wIE =0.05
g = 1             # global couping parameter
gamma = 0.641     # Only for excitatory in the synpatic equation
noi_sd =2      # Noise sd for the model
#noi_sd =0      # Noise sd for the model

tauE=0.1          # sec
tauI=0.01         # sec 
J=0.15            # nA  Effective NMDA conductance

# fI-curve parameters (Refer to Table 1 in Dermirtas 2019)

dE=0.16     # seconds 
dI=0.087
bE=125      # Hz
bI=177 
aE=310     #1/nC
aI=615

SC_scaled = g*J*(SC.T).T

# Define the fI curve
def fI(a,b,d,I):
	#print(max(I),min(I))
	#print(((a*I)-b)/(1-np.exp(-d*((a*I)-b))))
	return ((a*I)-b)/(1-np.exp(-d*((a*I)-b)))

# def status_message function
def status_message(txt="Progressing"):
	print("\n ++ "+ txt)

#-------------------------------------------------------------- 
# 5. Generate time series of current, rate and synaptic gating variables
#-------------------------------------------------------------- 

dt = 0.01  # [s]
T =  2     # [s] Simulation duration
t_plot = np.linspace(0, T, int(T/dt)+1)
nt = len(t_plot)
v=np.random.normal(0, noi_sd, size=(nt,nROI))   # IID noise matrix

# Synaptic variables are initilized here to solve for the steady state values
# The following two variables of size N_ROI X  1
# We learnt that initialization affect the final solution. By looking at the MSE we chose the initial values. The following initial values appears best for this problem.

SE_back= 0.5 * np.ones(nROI)
SI_back= 0.5 * np.ones(nROI)

# Noise need not be IID; it can have covariance matrix
# But in the Dermirtas the noise is considered IID across the population (See below Eqn. 46)

status_message("Solving for the steady-state solution")
# function to solve for the steady-state values

def solve4synaptic(s):
	sI=s[:nROI]
	sE=s[nROI:]
	I_inh = (wI*Ib) + (wEI * sE) - sI
	r_inh = fI(aI,bI,dI,I_inh)
	res1 = sI-(r_inh*tauI)
	
	longrange_E=np.dot(SC_scaled,sE)
	I_exc = (wE*Ib) + (wEE*sE)+(g*J*longrange_E) - (wIE*sI)
	r_exc = fI(aE,bE,dE,I_exc)
	res2= sE - (gamma*r_exc/((1/tauE)+(gamma*r_exc)))	

	return np.hstack((res1,res2))

x0=np.hstack((SI_back,SE_back))
opti_res=least_squares(solve4synaptic,x0,bounds=(0,1))
s=opti_res['x']  # Final solution

status_message("Result of steady state solution")
print("SOLUTION FOUND: ",opti_res['success'])
print("MSE : ",round(pow(opti_res['fun'],2).mean(),10))  #MSE

#-------------------------------------------------------------- 
# Rate, synpatic time series initializtion
# (N_time_point X N_ROI) matrix
#-------------------------------------------------------------- 

RE,RI =np.zeros((nt,nROI)),np.zeros((nt,nROI))
SE,SI=np.zeros((nt,nROI)),np.zeros((nt,nROI))
I_inh,I_exc =np.zeros((nt,nROI)),np.zeros((nt,nROI))
SI[0] = s[:nROI]
SE[0] = s[nROI:]


#-------------------------------------------------------------- 
# Running the network
#-------------------------------------------------------------- 

for t in range(1,nt):

	longrange_E = np.dot(SC_scaled,SE[t-1])
	
	#print("minimum lr",min(longrange_E))
	#print("maximum lr",max(longrange_E))

	# Inhibition
	I_inh[t-1] = (wI*Ib) + (wEI * SE[t-1]) - SI[t-1]
 	#          Back-ground    Excitatory   Inhibitory 
	RI[t-1] = fI(aI,bI,dI,I_inh[t-1])

	# Excitation
	I_exc[t-1] = (wE*Ib) + (wEE*SE[t-1])+ longrange_E - (wIE*SI[t-1])
 	#      Back-ground    Excitatory    Long-range     Inhibitory 
	RE[t-1] = fI(aE,bE,dE,I_exc[t-1])

	status_message("Timept {}".format(round(t_plot[t-1],2)))
	d={"RE_min": round(min(RE[t-1]),2),
	"RE_max": round(max(RE[t-1]),2),
	"RI_min": round(min(RI[t-1]),2),
	"RI_max": round(max(RI[t-1]),2)}
	print(d)

	# Synaptic gating variables
    
	d_SE = (-SE[t-1]/tauE) +((1-SE[t-1])*gamma*RE[t-1])+v[t-1]  
	d_SI = (-SI[t-1]/tauI) +RI[t-1]+ v[t-1] 
    
	SE[t] = np.clip(SE[t-1] + (d_SE * dt),0,1)         #First-order Euler method
	SI[t] = np.clip(SI[t-1] + (d_SI * dt),0,1)
	
	# Syaptic variable is clipped by the codes in Dermirtas	
	d={"SE_min": round(min(SE[t]),2),
	"SE_max": round(max(SE[t]),2),
	"SI_min": round(min(SI[t]),2),
	"SI_max": round(max(SI[t]),2)}
	print(d)



_,ax=plt.subplots(2,3,figsize=(20,5))
ax[0,0].plot(SE[:,0])
ax[0,1].plot(SE[:,1])
ax[0,2].plot(SE[:,20])
ax[1,0].plot(SE[:,50])
ax[1,1].plot(SE[:,120])
ax[1,2].plot(SE[:,220])
plt.show()

_,ax=plt.subplots(2,3,figsize=(20,5))
ax[0,0].plot(RE[:,0])
ax[0,1].plot(RE[:,1])
ax[0,2].plot(RE[:,20])
ax[1,0].plot(RE[:,50])
ax[1,1].plot(RE[:,120])
ax[1,2].plot(RE[:,220])
plt.show()



_,ax=plt.subplots(2,3,figsize=(20,5))
ax[0,0].plot(I_exc[:,0])
ax[0,1].plot(I_exc[:,1])
ax[0,2].plot(I_exc[:,20])
ax[1,0].plot(I_exc[:,50])
ax[1,1].plot(I_exc[:,120])
ax[1,2].plot(I_exc[:,220])
plt.show()

""" 


def plot_response(p,I_stim_exc,SE):

	area_name_list  = p['areas']
	area_idx_list   = [-1]+[p['areas'].index(name) for name in area_name_list]
	f, ax_list      = plt.subplots(len(area_idx_list), figsize=(7,30),sharex=True)

	for ax, area_idx in zip(ax_list,area_idx_list):
	    if area_idx < 0:
	      y_plot=I_stim_exc[:,area_stim_idx]
	      txt     = 'Input'
	    else:
	      y_plot  = SE[:,area_idx]
	      txt     = p['areas'][area_idx]

	    y_plot = y_plot - y_plot.min()
	    ax.plot(t_plot, y_plot)
	    ax.text(0.9, 0.6, txt, transform=ax.transAxes)

	    ax.set_yticks([y_plot.max()])
	    ax.set_yticklabels(['{:0.4f}'.format(y_plot.max())])
	    ax.spines["right"].set_visible(False)
	    ax.spines["top"].set_visible(False)
	    ax.xaxis.set_ticks_position('bottom')
	    ax.yaxis.set_ticks_position('left')

	f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical')
	ax.set_xlabel('Time (ms)')
	plt.show()


plot_response(p,I_stim_exc,SE)

#-------------------------------------------------------------- 
# Step 2b. Set-up Ballon-Model parameters
#-------------------------------------------------------------- 
# parameters used by Dermirtas (from Friston 2003 and Obata 2004)

rho=0.34    # resting-oxygen extraction fraction
alpha=0.32  # Grubb's exponent
Vo=0.02     # resting bllod volume fraction
gamma=0.41  # [s^-1]  rate of flow-dependent elimination
kappa=0.65  # [s^-1]  rate of signal decay
tau = 0.98  # [s]  hemodynamic transit time(from Obata 2004)  


k1=3.72
k2=0.53
k3=0.53

# Intialization of the variables
x=np.zeros((n,p['n_area']))    # vaso-dilatory signal
f=np.ones((n,p['n_area']))   # blood inflow
v=np.ones((n,p['n_area']))    # blood volume
q=np.ones((n,p['n_area']))    # deoxy hemoglobin 
y=np.zeros((n,p['n_area']))  # BOLD signal  

#-------------------------------------------------------------- 
# 6. Generate BOLD time series from rate or synaptic gating variables
#-------------------------------------------------------------- 

status_message("Beginning BOLD simulation")


#---------------------------------------------------------------------------------
# Running the Ballon model using the synaptic gating variable
#---------------------------------------------------------------------------------

for t in range(1, n):
    # x is Vaso-dilatory time series
    dx =  s[t-1] - (kappa*x[t-1])- (gamma *(f[t-1]-1))
    x[t]= x[t-1] + dx *dt

    # f blood inflow time series
    df =x[t-1]
    f[t] = f[t-1]+ df *dt

    # v is blood volume time series
    dv = (f[t-1] - pow(v[t-1],1/alpha))/tau
    v[t]=v[t-1] + dv *dt

    # q is deoxy hemoglobin time series
    dq = ((f[t-1]/rho)*(1-pow(1-rho,1/f[t-1])) - (q[t-1]*pow(v[t-1],(1-alpha)/alpha)))/tau
    q[t] = q[t-1] + dq*dt

    # y is BOLD time series
    y[t] = Vo*(k1*(1-q[t]) +  k2*(1-(q[t]/v[t])) + k3*(1-v[t]))


#-------------------------------------------------------------- 
# 4. Construction of random input for testing purpose
#-------------------------------------------------------------- 

Iext=1
"""
