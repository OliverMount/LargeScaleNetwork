import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pandas as pd

def F(I, a=0.27, b=108., d=0.154):
    """F(I) for vector I"""
    return (a*I - b)/(1.-np.exp(-d*(a*I - b)))

def status_message(txt="progressing"):
	print("\n ++ " + txt)

#-------------------------------------------------------------- 
# Network Parameters
#-------------------------------------------------------------- 
p={}

p['beta_exc']   = 0.066 # Hz/pA
p['beta_inh']   = 0.351 # Hz/pA
p['tau_exc']    = 60    # ms
p['tau_inh']    = 10    # ms
p['wEE']        = 250.2  # pA
p['wIE']        = 303.9  # pA
p['wEI']        = 8.11  # pA/Hz
p['wII']        = 12.5  # pA/Hz
p['muEE']       = 125.1  # pA/Hz
p['eta']        = 3.4
p['gamma']      = 0.641


fI = lambda x : x*(x>0) # f-I curve


#-------------------------------------------------------------- 
# Loading Data 
#-------------------------------------------------------------- 


dpath="/olive/Maths/data/"
FLN_MTX=pd.read_csv(dpath+"fln_mtx.csv")
FLN_MTX.index=FLN_MTX.columns
 
hier=pd.read_csv(dpath+"hier.csv",header=None)
hier.index=FLN_MTX.index
t=hier.iloc[:,0]/max(hier.iloc[:,0]) # Normalizing hierarchy to  (0     1)
hier.iloc[:,0]=t
#p['areas']=['V1','V4','7A','9/46d'] # for 4 areas 
#p['areas']=['V1','V4','7A','9/46d','8m','TEO','7A','9/46v','TEpd',    '24c']
p['areas']=list(FLN_MTX.columns) # for all 29 areas
p['n_area']=len(p['areas'])

print("The list of ROIs considered are")
print(p['areas'])
 
# Hierarchy information
p['hier_vals']=np.squeeze(np.array(hier.loc[p['areas']]))
 
# FLN  matrix
p['fln_mat']=np.array(FLN_MTX.loc[p['areas'],p['areas']])
p['exc_scale'] = (1+p['eta']*p['hier_vals'])



area_act = 'V1'

status_message('Running network with stimulation to ' + area_act)

#-------------------------------------------------------------- 
# Redefine Parameters
#-------------------------------------------------------------- 

# Definition of combined parameters

local_EE        =  p['wEE'] * p['exc_scale']
local_EI        = -p['wEI']
local_IE        =  p['beta_inh'] * p['wIE'] * p['exc_scale']
local_II        = -p['beta_inh'] * p['wII']

fln_scaled      = (p['exc_scale'] * p['fln_mat'].T).T

#-------------------------------------------------------------- 
# Simulation Parameters

dt              = 0.2   # ms
T               = 2500  # ms
t_plot          = np.linspace(0, T, int(T/dt)+1)
n             = len(t_plot)

I_bkg_exc = 400.0
I_bkg_inh = 61.76

# Solve for baseline gating variable and firing rates.
def _solver(s):
	r_inh = p['beta_inh'] * (p['exc_scale'] * p['wIE'] *s + I_bkg_inh / p['beta_inh']) / (1 + p['beta_inh']* p['wII'])
	longrange_E = np.dot(fln_scaled,s)
	r_exc = F(local_EE * s + local_EI * r_inh + p['muEE'] * longrange_E +I_bkg_exc)

	return s - p['gamma'] * (p['tau_exc'] /1000) * r_exc / (1 + p['gamma'] * (p['tau_exc'] /1000) * r_exc)


x0 = np.ones(p['n_area']) * 0.05
s_tgt = least_squares(_solver, x0, bounds=(np.zeros(p['n_area']),
											 np.ones(p['n_area'])))['x']
r_inh_tgt = p['beta_inh'] * (p['exc_scale'] * p['wIE'] *
							 s_tgt + I_bkg_inh /
							 p['beta_inh']) / (1 + p['beta_inh'] * p['wII'])

longrange_E = np.dot(fln_scaled,s_tgt)
I_exc       = local_EE*s_tgt + local_EI*r_inh_tgt + \
			   p['muEE'] * longrange_E + I_bkg_exc
r_exc_tgt = F(I_exc)

# Set stimulus input
I_stim_exc      = np.zeros((n,p['n_area']))
area_stim_idx   = p['areas'].index(area_act) # Index of stimulated area
time_idx        = (t_plot>100) & (t_plot<=350)
I_stim_exc[time_idx, area_stim_idx] = 200.0

#---------------------------------------------------------------------------------
# Declare and initialise  synatic and rate time series variables
#---------------------------------------------------------------------------------

s = np.zeros((n,p['n_area']))
r_exc = np.zeros((n,p['n_area']))
r_inh = np.zeros((n,p['n_area']))

# Set activity to background firing
s[0] = s_tgt
r_inh[0] = r_inh_tgt
r_exc[0] = r_exc_tgt

#---------------------------------------------------------------------------------
# Running the network
#---------------------------------------------------------------------------------

for t in range(1, n):

	d_s = -s[t-1] + p['gamma'] * (p['tau_exc']/1000) * (1-s[t-1]) * r_exc[t-1]
	s[t] = s[t-1] + d_s * dt/p['tau_exc']

	longrange_E = np.dot(fln_scaled,s[t])

	I_exc= local_EE*s[t] + local_EI*r_inh[t-1] +p['muEE'] * longrange_E + I_bkg_exc + I_stim_exc[t]

	I_inh= local_IE*s[t] + local_II*r_inh[t-1] + I_bkg_inh

	d_r_inh= -r_inh[t-1] + fI(I_inh)

	r_exc[t]= F(I_exc)
	r_inh[t]= r_inh[t-1] + d_r_inh * dt/p['tau_inh']

#---------------------------------------------------------------------------------
# Plotting the results
#---------------------------------------------------------------------------------

area_name_list  = p['areas']
area_idx_list   = [-1]+[p['areas'].index(name) for name in area_name_list]
f, ax_list      = plt.subplots(len(area_idx_list), sharex=True,figsize=(7,30))

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


#-------------------------------------------------------------- 
# Plotting the synaptic time series
#-------------------------------------------------------------- 

def plot_ts(s,ylabel="Amplitude"):
		area_name_list  = p['areas']
		area_idx_list   = [p['areas'].index(name) for name in area_name_list]
		f, ax_list      = plt.subplots(len(area_idx_list), sharex=True,figsize=(7,30))

		for ax, area_idx in zip(ax_list, area_idx_list):
			txt     = p['areas'][area_idx]

			y_plot =s[:,area_idx]        # Synaptic gating variable time series 
			ax.plot(t_plot, y_plot)
			ax.text(0.9, 0.6, txt, transform=ax.transAxes)

			ax.set_yticks([y_plot.max()])
			ax.set_yticklabels(['{:0.4f}'.format(y_plot.max())])
			ax.spines["right"].set_visible(False)
			ax.spines["top"].set_visible(False)
			ax.xaxis.set_ticks_position('bottom')
			ax.yaxis.set_ticks_position('left')

		f.text(0.01,0.5,ylabel,va='center', rotation='vertical')
		ax.set_xlabel('Time (ms)')
		plt.show()

plot_ts(s,"Synaptic gating variable")
#--------------------------------------------------------------
# Ballon model simulation starts here 
#--------------------------------------------------------------

status_message("Beginning BOLD simulation")


# parameters used by Dermirtas (from Friston 2003 and Obata 2004)

rho=0.34	# resting-oxygen extraction fraction
alpha=0.32	# Grubb's exponent
Vo=0.02		# resting bllod volume fraction
gamma=0.41  # [s^-1]  rate of flow-dependent elimination
kappa=0.65  # [s^-1]  rate of signal decay
tau = 0.98  # [s]  hemodynamic transit time(from Obata 2004)  


k1=3.72
k2=0.53
k3=0.53

# Intialization of the variables
x=np.zeros((n,p['n_area']))    # vaso-dilatory signal
f=np.ones((n,p['n_area']))	 # blood inflow
v=np.ones((n,p['n_area']))    # blood volume
q=np.ones((n,p['n_area']))    # de
y=np.zeros((n,p['n_area']))	 # BOLD signal	


#---------------------------------------------------------------------------------
# Running the Ballon model using the synaptic gating variable
#---------------------------------------------------------------------------------

for t in range(1, n):

	dx =  s[t-1] - (kappa*x[t-1])- (gamma *(f[t-1]-1))
	x[t]= x[t-1] + dx *dt
    
	df =x[t-1]
	f[t] = f[t-1]+ df *dt


	dv = (f[t-1] - pow(v[t-1],1/alpha))/tau
	v[t]=v[t-1] + dv *dt

	dq = ((f[t-1]/rho)*(1-pow(1-rho,1/f[t-1])) - (q[t-1]*pow(v[t-1],(1-alpha)/alpha)))/tau
	#print(f[t-1])
	q[t] = q[t-1] + dq*dt
	
	y[t] = Vo*(k1*(1-q[t]) +  k2*(1-(q[t]/v[t])) + k3*(1-v[t]))



plot_ts(x,"Vaso-dilatory signal")
plot_ts(f,"Blood inflow")
plot_ts(y,"BOLD response")

status_message("End of BOLD simulation")
print()
