# This program with inhibition control as metioned by Deco and Dermirtas

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

print(np.amax(SC))
print(np.amin(SC))
plt.imshow(SC,vmin=0.01,vmax=1)
plt.show()



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

wEE=0.01 
wEI=0.05
wII=0.04
#wIE =0.05         # This will be fixed later via Feedback inhibition control
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
T =  600   # [s] Simulation duration
#t_plot = np.linspace(0, T, int(T/dt)+1)
t_plot = np.arange(0,T,dt)
nt = len(t_plot)
v=np.random.normal(0, np.sqrt(dt)*noi_sd, size=(nt,nROI))   # IID noise matrix


# Noise need not be IID; it can have covariance matrix
# But in the Dermirtas the noise is considered IID across the population (See below Eqn. 46)

# Synaptic variables are initilized here to solve for the steady state values
# The following two variables of size N_ROI X  1
# We learnt that initialization affect the final solution. By looking at the MSE we chose the initial values. The following initial values appears best for this problem.

SI_back= 0.5 * np.ones(nROI) 
status_message("Solving for the steady-state solution")

# function to solve for the steady-state values

r_desired=3  # Hz Excitatory background  rate

status_message("Solving current (I) the background rate of ~" + str(r_desired)+ " Hz")

def invfI(r_desired,imin=0.1,imax=0.5,inc=0.0001):
	I=np.arange(imin,imax,inc)
	r=fI(aE,bE,dE,I)
	res=abs((r-r_desired))
	ind=np.argmin(res)
	return (res[ind],r[ind],I[ind])  # residual, rate and current

_,r_e,I_e=invfI(r_desired)
I_exc=I_e *np.ones(nROI)
print("Solved for the rate of",r_e)
# Solving for excitatory syanptic gating variable

se= (gamma*r_desired/((1/tauE)+(gamma*r_desired)))
sE= se * np.ones(nROI)   # All regions same 3 Hz

#print(invfI(3))	

def solve4synaptic(sI):
	I_inh = (wI*Ib) + (wEI * sE) - sI
	r_inh = fI(aI,bI,dI,I_inh)
	res = sI-(r_inh*tauI)
	return res

opti_res=least_squares(solve4synaptic,SI_back,bounds=(0,1))
sI=opti_res['x']  # Optimal steady-state inhibitory syanptic gating variable
#print("sI",sI)

wIE=-I_exc.mean() +(wE*Ib) + (wEE*sE.mean())+(g*J*sE.mean())/(sI.mean())  # Finding the inhibition control explicitly
#wIE=-I_exc.mean() +(wE*Ib) + (wEE*sE.mean())+(g*J*np.dot(SC_scaled,sE).mean())/(sI.mean())  # Finding the inhibition control explicitly

status_message("Result of steady state solution")
print("SOLUTION FOUND: ",opti_res['success'])
print("MSE : ",round(pow(opti_res['fun'],2).mean(),10))  #MSE
print("Optimal wIE for FIC is ", wIE)
print("Optimal mean sI", sI.mean())
print("Optimal sE",se)

#-------------------------------------------------------------- 
# Rate, synpatic time series initializtion
# (N_time_point X N_ROI) matrix
#-------------------------------------------------------------- 

RE,RI =np.zeros((nt,nROI)),np.zeros((nt,nROI))
SE,SI=np.zeros((nt,nROI)),np.zeros((nt,nROI))
I_inh,I_exc =np.zeros((nt,nROI)),np.zeros((nt,nROI))
SI[0] = sI
SE[0] = sE

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

	"""
	status_message("Timept {}".format(round(t_plot[t-1],2)))
	d={"RE_min": round(min(RE[t-1]),2),
	"RE_max": round(max(RE[t-1]),2),
	"RI_min": round(min(RI[t-1]),2),
	"RI_max": round(max(RI[t-1]),2)}
	print(d)
	"""
	# Synaptic gating variables
    
	d_SE = (-SE[t-1]/tauE) +((1-SE[t-1])*gamma*RE[t-1])+v[t-1]  
	d_SI = (-SI[t-1]/tauI) +RI[t-1]+ v[t-1] 
    
	SE[t] = np.clip(SE[t-1] + (d_SE * dt),0,1)         #First-order Euler method
	SI[t] = np.clip(SI[t-1] + (d_SI * dt),0,1)
	#SE[t] = SE[t-1] + (d_SE * dt)                       #First-order Euler method
	#SI[t] = SI[t-1] + (d_SI * dt)
	
	# Syaptic variable is clipped by the codes in Dermirtas
	# is it valid analytically?	
	"""
	d={"SE_min": round(min(SE[t]),2),
	"SE_max": round(max(SE[t]),2),
	"SI_min": round(min(SI[t]),2),
	"SI_max": round(max(SI[t]),2)}
	print(d)
	"""

def plot_bold(s,i=(0,1,20,50,120,220),txt="Synaptic gating variable",t_plot=t_plot,T=T):
		f,ax=plt.subplots(2,3,figsize=(20,5))
		#time=t_plot
		time=np.arange(0,s.shape[0])/(60/dt) 
		ax[0,0].plot(time,s[:,i[0]])
		ax[0,1].plot(time,s[:,i[1]])
		ax[0,2].plot(time,s[:,i[2]])
		ax[1,0].plot(time,s[:,i[3]])
		ax[1,1].plot(time,s[:,i[4]])
		ax[1,2].plot(time,s[:,i[5]])
		f.suptitle(txt)
		plt.show()

#plot_bold(SE,txt="Synaptic gating variable")
#plot_bold(RE,txt="Rate time series")
#plot_bold(I_exc,txt="Excitatory current time series")

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


#-------------------------------------------------------------- 
# 6. Generate BOLD time series from rate or synaptic gating variables
#-------------------------------------------------------------- 

status_message("Beginning BOLD simulation")

s=SE[6000:]   # Excitatory synaptic gating variables are used

#print(s.shape)
#O=np.ones((2000,nROI))
#Z=np.zeros((nt-2000,nROI))
#s=np.vstack((O,Z))

# Intialization of the variables
nt=s.shape[0]
x=np.ones((nt,nROI))    # vaso-dilatory signal
f=np.ones((nt,nROI))   # blood inflow
v=np.ones((nt,nROI))    # blood volume
q=np.ones((nt,nROI))    # deoxy hemoglobin 
y=0.03*np.ones((nt,nROI))  # BOLD signal 

 
#---------------------------------------------------------------------------------
# Running the Ballon model using the synaptic gating variable
#---------------------------------------------------------------------------------
for t in range(1,s.shape[0]):
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
    dq = ((f[t-1]/rho)*(1-pow(1-rho,(1/f[t-1]))) - (q[t-1]*pow(v[t-1],(1-alpha)/alpha)))/tau
    q[t] = q[t-1] + dq*dt

    # y is BOLD signal change time series
    y[t] = Vo*(k1*(1-q[t]) +  k2*(1-(q[t]/v[t])) + k3*(1-v[t]))




plot_bold(s,txt="Synaptic variables")
#plot_bold(x[:-1000],txt="Vaso-dilatory")
#plot_bold(f[:-1000:],txt="blood in-flow")
#plot_bold(v[:-1000:],txt="blood volume")
#plot_bold(q[:-1000:,],txt="deoxy hemoglobin")
plot_bold(y[6000:]*100,txt="Percent BOLD change")


# Resampling BOLD signal
#from scipy import signal

#secs = len(y)   

#for l in range(nROI):
	
