"""
@author: olive following the codes of Chauduri
"""

#------------------------------------------------------------------------------ 
# Necessary modules
#------------------------------------------------------------------------------ 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import normal as rnorm
 
dpath="/olive/Maths/data/"

#from Rate_Modules.modules import *
#------------------------------------------------------------------------------ 
# Network Parameters
#------------------------------------------------------------------------------ 
p={}
p['beta_exc'] = 0.066  # Hz/pA
p['beta_inh'] = 0.351  # Hz/pA
p['tau_exc']=20 # ms
#p['tau_exc'] = np.array([20,50,100,200])  # ms
#p['tau_exc'] = np.array([20,20,20,20])  # ms
#p['tau_exc']=20
p['tau_inh'] = 10  # ms
p['wEE'] = 24.3  # pA/Hz
p['wIE'] = 12.2  # pA/Hz
p['wEI'] = 19.7  # pA/Hz
p['wII'] = 12.5  # pA/Hz
p['muEE'] = 33.7  # pA/Hz
p['muIE'] = 25.3  # pA/Hz
p['eta'] = 0.68

def print_line(n=20):
    print("-"*n)	


## We got all the information of the hierarchy values and the fln matrix.
# Let me use the original values from the author webpage

FLN_MTX=pd.read_csv(dpath+"fln_mtx.csv")
FLN_MTX.index=FLN_MTX.columns

print(" ROIs considered in this project")
print(FLN_MTX.columns)

hier=pd.read_csv(dpath+"hier.csv",header=None)
hier.index=FLN_MTX.index

t=hier.iloc[:,0]/max(hier.iloc[:,0]) # Normalizing hierarchy to  (0 1)
hier.iloc[:,0]=t


#p['areas']=['V1','V4','7A','9/46d'] # for 4 areas 
#p['areas']=['V1','V4','7A','9/46d','8m','TEO','7A','9/46v','TEpd','24c']
p['areas']=list(FLN_MTX.columns) # for all 29 areas
p['n_area']=len(p['areas'])

# Hierarchy information
p['hier_vals']=np.squeeze(np.array(hier.loc[p['areas']]))

# FLN  matrix
p['fln_mat']=np.array(FLN_MTX.loc[p['areas'],p['areas']])
p['exc_scale'] = (1+p['eta']*p['hier_vals'])


# plt.imshow(np.log(p['fln_mat']))
# Sign function
fI = lambda x : x*(x>0) # f-I curve

#-----------------------------------------------------------
# Generate random rate time series for each ROI
#----------------------------------------------------------

########### Choose the injection area
area_act = 'V1'
print('Running network with stimulation to ' + area_act)

 
# Definition of combined parameters

local_EE = p['beta_exc'] * p['wEE'] * p['exc_scale']
local_EI = -p['beta_exc'] * p['wEI']
local_IE =  p['beta_inh'] * p['wIE'] * p['exc_scale']
local_II = -p['beta_inh'] * p['wII']

fln_scaled = (p['exc_scale'] * p['fln_mat'].T).T



#---------------------------------------------------------------------------------
# Simulation Parameters
#---------------------------------------------------------------------------------


dt = 0.2  # ms
T = 2500  # ms
t_plot = np.linspace(0, T, int(T/dt)+1)
n_t = len(t_plot)

E_back=10  # Back-ground rate for excitation
I_back=35  # Back-ground rate for inhibition

# From target background firing inverts background inputs
r_exc_tgt = E_back * np.ones(p['n_area'])
r_inh_tgt = I_back * np.ones(p['n_area'])

longrange_E = np.dot(fln_scaled,r_exc_tgt)
I_bkg_exc = r_exc_tgt - (local_EE*r_exc_tgt + local_EI*r_inh_tgt
                         + p['beta_exc']*p['muEE']*longrange_E)
I_bkg_inh = r_inh_tgt - (local_IE*r_exc_tgt + local_II*r_inh_tgt
                         + p['beta_inh']*p['muIE']*longrange_E)

#--------------------------------------------------------------
# Set white noise parameters
#--------------------------------------------------------------
me=0
SD_1=0.5 # in Hz
SD_2=0.00001

I_stim_exc = np.zeros((n_t,p['n_area']))
area_stim_idx = p['areas'].index(area_act) # Index of stimulated area
area_no_stim=tuple([i for i in range(p['n_area']) if i!= area_stim_idx])
I_stim_exc[:,area_stim_idx] = rnorm(me,SD_1,n_t)
I_stim_exc[:,area_no_stim] = rnorm(me,SD_2,p['n_area']-1)

#---------------------------------------------------------------------------------
# Final rate time series (n_timepoints X N_roi)
#---------------------------------------------------------------------------------

r_exc = np.zeros((n_t,p['n_area']))
r_inh = np.zeros((n_t,p['n_area']))

#---------------------------------------------------------------------------------
# Initialization
#---------------------------------------------------------------------------------

# Set activity to background firing
r_exc[0] = r_exc_tgt
r_inh[0] = r_inh_tgt

#---------------------------------------------------------------------------------
# Running the network
#---------------------------------------------------------------------------------

for t in range(1, n_t):
    longrange_E = np.dot(fln_scaled,r_exc[t-1])
    I_exc = (local_EE*r_exc[t-1] + local_EI*r_inh[t-1] +
             p['beta_exc'] * p['muEE'] * longrange_E +
             I_bkg_exc + I_stim_exc[t])

    I_inh = (local_IE*r_exc[t-1] + local_II*r_inh[t-1] +
             p['beta_inh'] * p['muIE'] * longrange_E + I_bkg_inh)

    d_r_exc = -r_exc[t-1] +  fI(I_exc)
    d_r_inh = -r_inh[t-1] +  fI(I_inh)

    r_exc[t] = r_exc[t-1] + d_r_exc * dt/p['tau_exc']
    r_inh[t] = r_inh[t-1] + d_r_inh * dt/p['tau_inh']


#-----------------------------------------------------------------------
#     Plotting the time series results
#-----------------------------------------------------------------------

area_name_list  = p['areas']
area_idx_list   = [-1]+[p['areas'].index(name) for name in area_name_list]
f, ax_list      = plt.subplots(len(area_idx_list), figsize=(10,30),sharex=True)

for ax, area_idx in zip(ax_list, area_idx_list):
    if area_idx < 0:
        y_plot  = I_stim_exc[:, area_stim_idx]
        txt     = 'Input'
    else:
        y_plot  = r_exc[:,area_idx]
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


#--------------------------------------------------------------------------------
#  PLOTTING RESULTS  
#-------------------------------------------------------------------------------

# Neural rate series plots

area_name_list  = p['areas']
area_idx_list   = [-1]+[p['areas'].index(name) for name in area_name_list]
f, ax_list      = plt.subplots(len(area_idx_list), sharex=True)

for ax, area_idx in zip(ax_list, area_idx_list):
    if area_idx < 0:
        y_plot  = I_stim_exc[:, area_stim_idx]
        txt     = 'Input'
    else:
        y_plot  = r_exc[:,area_idx]
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

#########################################################################
### Autocorrelation calculation, plots and exponential fits ############# 
#########################################################################
## ACF of the ROIs are stacked as columns of numpy array

import statsmodels.api as sm             
from statsmodels.tsa.stattools import acf   # For autocorrelation
from  scipy.optimize import curve_fit       # For exponential curve fitting  


# Single exponential fit
def monoExp(x, tau):
    return  np.exp(-tau*x) 
 

_ = plt.figure(figsize=(10,8)) 
nl=1000                # Lag index for autocorrelation

ACF=np.zeros((nl,p['n_area']))
m=np.zeros(p['n_area'])
Tau_esti=np.zeros(p['n_area'])

#cols=['r','g','b','k']   # Colors for the plot
cols=['r']
para=(30)    # Initial value for optimization

for k in range(p['n_area']):
    ACF[:,k]=acf(r_exc[:,k], nlags=nl-1)
    plt.plot(np.arange(nl)*dt,ACF[:,k],cols[0],label=p['areas'][k])
    
    # Cuve fitting
    params,_ =  curve_fit(monoExp, np.arange(nl)*dt,ACF[:,k],para) 
    #m[k]=params[0]
    Tau_esti[k]=params[0] 
plt.legend()
plt.xlim(np.array([0, nl])*dt)
plt.title("Autocorrelation of rate changes at different regions",size=20)
plt.xlabel("Lags (msec)",size=14)
plt.ylabel("Normalized Autocorrelation",size=14)
plt.show()
 

_ = plt.figure(figsize=(12,10))  
for k in range(p['n_area']):
    plt.subplot(int(str(66)+str(k+1)))
    plt.plot(np.arange(nl)*dt,ACF[:,k],label="ACF data")
    plt.plot(np.arange(nl)*dt, 
             monoExp(np.arange(nl)*dt,Tau_esti[k]),
             '--', label="fitted")
    plt.title(p['areas'][k] + "-- Esti. Tau: "+ 
              str(round(1/Tau_esti[k],2)) + " msec" ,size=20)

plt.show()
################  Creation of BOLD resting state from the neural signals #####


# Hemodynamic function

def Hemodynamic(n,TR,tauh=1.25*1e3,d=2.25*1e3):
    # f=[]
    # for k in range(n):
    #     f.append((((k*TR)-d)*np.exp(((k*TR)-d)/tauh))/tauh**2)

    return  [(((k*TR)-d)*np.exp(-((k*TR)-d)/tauh))/tauh**2  for k in range(n) if k!=0]
        
plt.plot(Hemodynamic(100,2))

############ COMPUTATION OF functional connectivity matrix ############

# def AUTOcorr(x,lags=10):
#     M =len(x)
#     r =np.zeros(M)  # One-sided autocorrelation  
#     for i in range(M): 
#         r[i]=(1/(M-i))*(sum(x[0:(M-i)] * x[i:M]))    # Dot product in r
  
#     return(r[0:(lags+1)])
   

# lgs=1000
# AC1= AUTOcorr(r_exc[:,0],lgs)
# AC2=AUTOcorr(r_exc[:,1],lgs)
# plt.plot(AC1)
# plt.plot(AC2/max(AC2))
# plt.show()
 
# lgs=1000
# autocorrelation = np.correlate(r_exc[:,0], r_exc[:,0], mode="full")
# sm.graphics.tsa.plot_acf(r_exc[:,0], lags=lgs)
# sm.graphics.tsa.plot_acf(r_exc[:,1], lags=lgs)
# sm.graphics.tsa.plot_acf(r_exc[:,2], lags=lgs)

#import matplotlib 
#matplotlib.pyplot.xcorr(r_exc[:,1], r_exc[:,1], normed=True, maxlags=1000)

# plt.plot(np.arange(nl)*dt,acf(r_exc[:,0], nlags=nl-1),'r',label=p['areas'][0])
# plt.plot(np.arange(nl)*dt,acf(r_exc[:,1], nlags=nl-1),'g',label=p['areas'][1])
# plt.plot(np.arange(nl)*dt,acf(r_exc[:,2], nlags=nl-1),'b',label=p['areas'][2])
# plt.plot(np.arange(nl)*dt,acf(r_exc[:,3], nlags=nl-1),'k',label=p['areas'][3])
# plt.legend()
# plt.xlim(np.array([0, nl])*dt)
# plt.title("Autocorrelation of rate changes at different regions",size=20)
#plt.ylim([0,1.1])


