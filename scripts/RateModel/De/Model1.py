#-------------------------------------------------------------- 
# Load necessary modules
#-------------------------------------------------------------- 

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import h5py
#from ratemodels import *
#from data import *

#--------------------------------------------------------------
# Steps of rate model
#--------------------------------------------------------------

#--------------------------------------------------------------
# Step1. Load the data 
#--------------------------------------------------------------

ipdir = "/olive/Maths/R/De/data/"
opdir ="/olive/Maths/R/De/output/" 

fname= ipdir+"demirtas_neuron_2019.hdf5"

with h5py.File(fname,"r") as f:
    # List all groups
    KEYS=f.keys()	
    print("Keys: %s" % KEYS)
    SC = f['sc'][()]             # 1a. Load Strctural connectivity
    T1WT2W = f['t1wt2w'][()]     # 1b. Load  T1w/T2W  

# Normalise the SC here



#-------------------------------------------------------------- 
# Step2a. Set up the rate-model parameters
#-------------------------------------------------------------- 

nROI=SC.shape[0]

# fixed model weights here

wE=1   # Weights for the background current
wI=0.7
Ib = 0.382        # nA

wEE=0.8 
WEI=0.5
wII=0.4
wIE =0.4
g = 1             # global couping parameter
gamma = 0.641     # Only for excitatory in the synpatic equation
noi_sd =0.34      # Noise sd for the model


JNMDA=1
tauE=0.1          # sec
tauI=0.01         # sec 
J=0.15            # nA  Effective NMDA conductance

# fI-curve parameters (Refer to Table 1 in Dermirtas 2019)

dE=0.16     # seconds 
dI=0.087
bE=125      # Hz
bI=177 
aE=310      #1/nC
aI=615

# Define the fI curve
def fI(a,b,d,I):
	"""
	This function returns the fI curve for given parameters
	"""
	r=[((a*l)-b)/(1-np.exp(-d*((a*l)-b))) for l in I]
	return np.array(r) 
	
	


#-------------------------------------------------------------- 
# Step 2b. Set-up Ballon-Model parameters
#-------------------------------------------------------------- 




#-------------------------------------------------------------- 
# Step3. Optimization parameters if any 
#-------------------------------------------------------------- 

#-------------------------------------------------------------- 
# 4. Construction of random input for testing purpose
#-------------------------------------------------------------- 

Iext=1


#-------------------------------------------------------------- 
# 5. Generate time series of current, rate and synaptic gating variables
#-------------------------------------------------------------- 

dt = 0.2  # ms
T = 2500  # ms
t_plot = np.linspace(0, T, int(T/dt)+1)
n_t = len(t_plot)

E_back=3  # Back-ground rate for excitation
I_back=3  # Back-ground rate for inhibition

# From target background firing inverts background inputs
# The following two variables of size N_ROI X  1
r_exc_tgt = E_back * np.ones(nROI)
r_inh_tgt = I_back * np.ones(nROI)
#-------------------------------------------------------------- 
# Final rate time series initializtio  (N_time_point X N_ROI) matrix
#-------------------------------------------------------------- 
r_exc = np.zeros((n_t,nROI))
r_inh = np.zeros((n_t,nROI))

# Set activity to background firing at the initial time point (why?)
r_exc[0] = r_exc_tgt
r_inh[0] = r_inh_tgt


# Definition of combined parameters
local_EE =  wEE 
local_EI = - wEI
local_IE =  wIE 
local_II = -wII
SC_scaled = g*(SC.T).T

longrange_E = np.dot(SC_scaled,r_exc_tgt)

I_bkg_exc = r_exc_tgt - (wEE*r_exc_tgt + wEI*r_inh_tgt
                         + longrange_E)

I_bkg_inh = r_inh_tgt - (wIE*r_exc_tgt + wII*r_inh_tgt
                         + longrange_E)
#-------------------------------------------------------------- 
# Stimulus
#-------------------------------------------------------------- 

# Set stimulus input (N_time_points X N_ROI)
# Here zero stimuls is given in other areas so that back ground activity dominates in those areas)
I_stim_exc = np.zeros((n_t,nROI))

area_stim_idx = ROInames.index(area_act) # Index of stimulated area
time_idx = (t_plot>100) & (t_plot<=350)
I_stim_exc[time_idx, area_stim_idx] = 41.8646  # in nA
#I_stim_exc[time_idx, area_stim_idx] = 0  # in nA
# Above value chosen so that V1 is driven up to 100 Hz

#-------------------------------------------------------------- 
# Running the network
#-------------------------------------------------------------- 

for t in range(1, n_t):
    longrange_E = np.dot(fln_scaled,r_exc[t-1])
    I_exc = (local_EE*r_exc[t-1] +  						# Excitatory
	     	local_EI*r_inh[t-1] +				    		# Inhibitory
            longrange_E + 	                             	# Long-range
            wE*I_bkg_exc 									# Back-ground excitatory

    I_inh = (local_IE*r_exc[t-1] + 				# Excitatory
	    	local_II*r_inh[t-1] +				# Inhibitory
	    	wI*I_bkg_inh						# Back-ground inhibitory  (We note here that no external input for ihbihitory neurons)

    d_r_exc = -r_exc[t-1] +  fI(I_exc)
    d_r_inh = -r_inh[t-1] +  fI(I_inh)

    r_exc[t] = r_exc[t-1] + d_r_exc * dt/p['tau_exc']
    r_inh[t] = r_inh[t-1] + d_r_inh * dt/p['tau_inh']



def plot_response(p,I_stim_exc,r_exc):

	area_name_list  = p['areas']
	area_idx_list   = [-1]+[p['areas'].index(name) for name in area_name_list]
	f, ax_list      = plt.subplots(len(area_idx_list), figsize=(7,30),sharex=True)

	for ax, area_idx in zip(ax_list,area_idx_list):
	    if area_idx < 0:
	      y_plot=I_stim_exc[:,area_stim_idx]
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


plot_response(p,I_stim_exc,r_exc)

#-------------------------------------------------------------- 
# 6. Generate BOLD time series from rate or synaptic gating variables
#-------------------------------------------------------------- 

# Implementation of Ballon Model




#-------------------------------------------------------------- 
# 7. Finding functional connectivity of the model and the empirical data
#-------------------------------------------------------------- 




#-------------------------------------------------------------- 
# 8. Visualise the parameters
#-------------------------------------------------------------- 
