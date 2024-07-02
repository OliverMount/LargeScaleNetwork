#-------------------------------------------------------------- 
# Necessary modules
#--------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

## We got all the information of the hierarchy values and the fln matrix.
# Let me use the original values from the author webpage

FLN_MTX=pd.read_csv(dpath+"fln_mtx.csv")
FLN_MTX.index=FLN_MTX.columns

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

#plt.imshow(np.log(p['fln_mat']))
# Sign function
fI = lambda x : x*(x>0) # f-I curve


#-------------------------------------------------------------------------------
# Choose the injection area
#-------------------------------------------------------------------------------
area_act = 'V1'  # This name should be one of the ROI names in p['areas']
print('Running network with stimulation to ' + area_act)

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
# The following two variables of size N_ROI X  1
r_exc_tgt = E_back * np.ones(p['n_area'])
r_inh_tgt = I_back * np.ones(p['n_area'])

#---------------------------------------------------------------------------------
# Final rate time series initializtio  (N_time_point X N_ROI) matrix
#---------------------------------------------------------------------------------

r_exc = np.zeros((n_t,p['n_area']))
r_inh = np.zeros((n_t,p['n_area']))

# Set activity to background firing at the initial time point (why?)
# 
r_exc[0] = r_exc_tgt
r_inh[0] = r_inh_tgt


# Definition of combined parameters
local_EE = p['beta_exc'] * p['wEE'] * p['exc_scale']
local_EI = -p['beta_exc'] * p['wEI']
local_IE =  p['beta_inh'] * p['wIE'] * p['exc_scale']
local_II = -p['beta_inh'] * p['wII']
fln_scaled = (p['exc_scale'] * p['fln_mat'].T).T

longrange_E = np.dot(fln_scaled,r_exc_tgt)


#The purpose of the following statements are
# 1. It is without the input (so then just the effect of long range back ground influence in each area.
# 2. It quanifies how much background current in each area is use to the influence of the 
# background in other areas (The influence here is from within the area and from long range) 
# The following two variables of size N_ROI X  1
# This looks like this is stead-state values obtained by setting 
# dv/dt=0 (excluding the Iext) and what we obtained below is background rate

I_bkg_exc = r_exc_tgt - (local_EE*r_exc_tgt + local_EI*r_inh_tgt
                         + p['beta_exc']*p['muEE']*longrange_E)

temp=local_EE*r_exc_tgt + local_EI*r_inh_tgt+p['beta_exc']*p['muEE']*longrange_E
print("The background inhibitory current") 
print(temp)


I_bkg_inh = r_inh_tgt - (local_IE*r_exc_tgt + local_II*r_inh_tgt
                         + p['beta_inh']*p['muIE']*longrange_E)

print("The background excitatory current") 
print(I_bkg_exc)
print("The background inhibitory current") 
print(I_bkg_inh)
#-----------------------------------------------------------------------------
# Stimulus
#---------------------------------------------------------------------------

# Set stimulus input (N_time_points X N_ROI)
# Here zero stimuls is given in other areas so that back ground activity dominates in those areas)
I_stim_exc = np.zeros((n_t,p['n_area']))

area_stim_idx = p['areas'].index(area_act) # Index of stimulated area
time_idx = (t_plot>100) & (t_plot<=350)
I_stim_exc[time_idx, area_stim_idx] = 41.8646  # in nA
#I_stim_exc[time_idx, area_stim_idx] = 0  # in nA
# Above value chosen so that V1 is driven up to 100 Hz


#---------------------------------------------------------------------------------
# Running the network
#---------------------------------------------------------------------------------

for t in range(1, n_t):
    longrange_E = np.dot(fln_scaled,r_exc[t-1])
    I_exc = (local_EE*r_exc[t-1] +  				# Excitatory
	     local_EI*r_inh[t-1] +				# Inhibitory
             p['beta_exc'] * p['muEE'] * longrange_E + 		# Long-range
             I_bkg_exc +					# Back-ground excitatory
	     I_stim_exc[t])					# External Stimulus

    I_inh = (local_IE*r_exc[t-1] + 				# Excitatory
	    local_II*r_inh[t-1] +				# Inhibitory
	    I_bkg_inh +						# Back-ground inhibitory  (We note here that no external input for ihbihitory neurons)
            p['beta_inh'] * p['muIE'] * longrange_E)		# Long-range

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
