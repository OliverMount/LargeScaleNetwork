# Loading necessary modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter,filtfilt  # For temporal filtering
from scipy.stats import pearsonr				  # For calulating r, and pvalue
from scipy.optimize import least_squares		  # For solving the FIC problem	
import scipy.io as read   # for reading matlab files
import h5py
import mat73 as mat       # Very useful package if the matlab files are not loaded properly


dpath="/olive/Maths/R/Do/InputData/"   # Data path

# Define necessary functions here like fI curve  and status_message
def fI(a,b,d,I):
	#print(max(I),min(I))
	#print(((a*I)-b)/(1-np.exp(-d*((a*I)-b))))
	return ((a*I)-b)/(1-np.exp(-d*((a*I)-b)))

# def status_message function
def status_message(txt="Progressing"):
	print("\n ++ "+ txt)

#-------------------------------------------------------------- 
# Loading data
#-------------------------------------------------------------- 

nROI=68    # Number of ROI
NoS=389	   # Number of participants	
TR=0.754   # [s] HCP data TR
NoT=616   # Length of time series data
#--------------------------------------------------------------
# Gene data loading and processing
#--------------------------------------------------------------


Genes_selected=read.loadmat(dpath+"DKcortex_selectedGenes.mat")
#print(Genes_selected.keys())

# 27 genes and 34 ROI
gene= Genes_selected['expMeasures']

ratioE = np.zeros((nROI))
Coef_E=np.sum(gene[:,17:25],axis=1)     # 18:21 ampa+ 22:25 nmda/gaba
ratioE[:34]=Coef_E/Coef_E.max()
ratioE[34:]=ratioE[:34]                 # Assuming symmetry

ratioI = np.zeros((nROI))
sli=np.hstack((range(1,9),range(11,14)))
Coef_I=np.sum(gene[:,sli],axis=1)     # 18:21 ampa+ 22:25 nmda/gaba
ratioI[:34]=Coef_I/Coef_I.max()
ratioI[34:]=ratioI[:34]                 # Assuming symmetry

ratio=ratioE/ratioI
ratio=ratio/(max(ratio)-min(ratio))
ratio=ratio-max(ratio)+1                 # This value is verified

print("Shape of gene data: ", ratio.shape)  # 34 X 27
#print(ratio)
# Meylin data

Myelin=read.loadmat(dpath+"myelin_HCP_dk68.mat")
print("Shape of myelin data:",Myelin['t1t2Cortex'].shape)

#--------------------------------------------------------------
# Structural connectivity data and Normalisation
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

#--------------------------------------------------------------
# HCP time series data
#--------------------------------------------------------------

Data=read.loadmat(dpath+"DKatlas_timeseries.mat")
dat=Data['ts']
data=dat[:,sli,:]
#print("Shape of the HCP data from Deco. is :", dat.shape)
print("Shape of the HCP data after ROI selection is :", data.shape)

# NoT X nROI X NoS
#plt.plot(data[:,4,0])
#plt.show()
print("\n")


with open(dpath+"Filtered.npy",'rb') as f:
	fdata= np.load(f)          # Filtered data loading
	print("Done with loading the filtered data")
print("Shape of the filtered data is ", fdata.shape)

with open(dpath+"FCemp.npy",'rb') as f:
	FCemp=np.load(f)
	print("Empirical connectivity loaded properly")

print("Shape of the empirical data",FCemp.shape)

#with open(dpath+"FCemp_pvals.npy",'rb') as f:
#	FCemp_pvals=np.load(f)
#	print("P-values of Empirical connectivity loaded properly")

status_message("Group averaging functional connectivity")
FCemp=FCemp.mean(axis=2)
print("Shape of the group averaged empirical data",FCemp.shape)

plt.imshow(FCemp,cmap='jet')
plt.colorbar()
plt.show()



status_message("Setting the parameters of mean-field model")
#--------------------------------------------------------------
# Parameters for the mean field model
#-------------------------------------------------------------- 

dt=0.001   #Sampling rate of simulated neuronal activity (seconds)

G=2.1  	  # Optimised value for homogeneous model (used for heterogeneous model as well)

tauE=0.1  # [s]   NMDA
tauI=0.01 # [s]   GABA
gamma=0.641  
noi_sd=0.01  
JN=0.15  #[nA] NMDA conductance
I0=0.382  #[nA] Background current

#Iext External current in each population, exist only in task, that is, to compute ignition capacity and autocorrelation (for time constant measurement), which clearly needs Iext. For FC fitting, this parameters 
w=1.4 
wE=1      # for scaling the background (I0)
wI=0.7

# parameters for fI curve

aE=310   #[nC-1] Slope parameter
bE=125   #[Hz]  Threshold current above which linear fI begins (Wrong in deco; Not current)
dE=0.16  # shape of the curvature about b

aI=615  #[nC-1]
bI=177  #[Hz]
dI=0.087

# Gain factor that includes the gene information

#B=-0.3
#Z= 2    # These values are taken from their graph
#M=1+B+(Z*ratio)

M=1 # For homogenous model

# Implementing Feedback Inhibition Control
# This is done only once before generating the neural time series

r_desired=3  # Hz Excitatory background  rate
status_message("Solving excitatory current (Ie) for the background rate of ~" + str(r_desired)+ " Hz")

def invfI(r_desired,imin=0.3,imax=0.5,inc=0.00001):  # Please remember this range is choosen so that r in 3 Hz
	I=np.arange(imin,imax,inc)
	r=fI(aE,bE,dE,I)
	res=abs((r-r_desired))
	ind=np.argmin(res)
	return (res[ind],r[ind],I[ind])  # residual, rate and current

_,r_e,I_e=invfI(r_desired)
I_exc=I_e *np.ones(nROI)
print("The current in nA for the desired rate of ", round(r_desired,2) , "is ",round(I_e,4), "nA")


status_message("Solving inhibitory current (Ii) for FIC")
# Solving for excitatory syanptic gating variable

se= (gamma*r_desired/((1/tauE)+(gamma*r_desired))) 
status_message("The excitatory synaptic gating variable for the desired rate is "+str(se))
sE= se * np.ones(nROI)   # All regions same 3 Hz  Vector of nROI length

def solve4_inh_synaptic_current(inh):
	res =np.array([(wI*I0) + (JN * se)-tauI*fI(aI,bI,dI,k)-k  for k in inh])
	return res   # This is the optimal Inhitory current 
	
Ii_initial = 0.1
opti_res=least_squares(solve4_inh_synaptic_current,Ii_initial,bounds=(0,1))  # Bound depends on the backgound rate
Ii_optimal=opti_res['x']  # Optimal steady-state inhibitory syanptic gating variable

tol=0.001

MSE=round(pow(opti_res['fun'],2).mean(),10)
status_message("Result of steady state solution")
print("SOLUTION FOUND: ",opti_res['success'])
print("MSE : ",MSE)  #MSE
if(MSE>tol):
	status_message("** WARNING  ** Your FIC may be wrong! Check your FIC optimization ")
print("The inhibitory current (Ii) is needed for the FIC is", float(Ii_optimal), " nA")
status_message("Done with FIC calculations")

# Balanced weight for inhibition control 
si=tauI*fI(aI,bI,dI,Ii_optimal) 
w_fic_nsc=(-I_e +(wE*I0) + (w*se)+(G*JN*se))/(si)  				    # Without accounting for SC (what Dermirtas did)
w_fic=(-I_e +(wE*I0) + (w*se)+(G*JN*np.dot(C,sE)))/(si)  	# With accounting for SC  (what Deco did)

## w_fic from deco is
w_fic=np.array([1.25725461184159,1.18322718600971,
1.71775554129727,
1.37792439545450,
1.02991710250104,
1.59032700497003,
1.97386293282030,
1.66198810840158,
1.38700238932414,
1.80050064594701,
1.33395060228658,
1.53427496634519,
1.24743707085485,
1.66209622507771,
1.12473369010547,
1.42263423336833,
1.54954122094101,
1.15260930471495,
1.31959316452764,
1.49838129580934,
1.65107313466297,
1.30891078009504,
2.16666877185468,
1.87669329856464,
1.14537818871983,
1.97311997329518,
2.85238173517954,
2.36764777477341,
1.61637459640469,
1.78279899048766,
1.04060542573697,
1.01321947167343,
1.10180701596273,
1.56978757749162,
1.20424461101822,
1.24187204526054,
1.63793421752711,
1.35444708764854,
1.02388510563507,
1.47164297586351,
2.08717530048410,
1.57222812110898,
1.33295669025123,
1.83654476616739,
1.38330053524428,
1.54036635124452,
1.30485960255737,
1.66872316821639,
1.12753619617235,
1.45927375669247,
1.43649585718716,
1.20916331067406,
1.36973304829230,
1.45470200637611,
1.64685558066870,
1.29916992440682,
2.10193316446192,
1.79778661282215,
1.18664961177874,
1.99027748068886,
2.81813128584384,
2.45779165240477,
1.57029877140960,
1.69580450594884,
1.05772379793943,
1.01771773450844,
1.04792477243391,
1.63837436039589])

print("Deco",w_fic)

#status_message("Optimal weight for FIC (with SC) is " + str(w_fic_nsc))
#status_message("Optimal weight for FIC (without SC) is " + str(w_fic))

#wIE=wIE_nsc
#-------------------------------------------------------------- 
# Rate, synpatic time series initializtion
# (N_time_point X N_ROI) matrix
#-------------------------------------------------------------- 

sI= si*np.ones((nROI))
IE=I_e *np.ones(nROI)

status_message(" Intialized and beginning neural time series simulation")

#-------------------------------------------------------------- 
# 5. Generate time series of current, rate and synaptic gating variables
#-------------------------------------------------------------- 

T= NoT*TR   # [s] Total simulation time  seconds 
t_plot = np.arange(0,T,dt)
nt = len(t_plot)


vE=np.random.normal(0, np.sqrt(0.1)*noi_sd, size=(nt,nROI))   # Excitatory IID noise matrix
vI=np.random.normal(0,np.sqrt(0.1)*noi_sd,size=(nt,nROI))     # Inhibitory

RE,RI =np.zeros((nt,nROI)),np.zeros((nt,nROI))
SE,SI=np.zeros((nt,nROI)),np.zeros((nt,nROI))
I_inh,I_exc =np.zeros((nt,nROI)),np.zeros((nt,nROI))

#SI[0] = sI
#SE[0] = sE

SI[0]=0.001*np.ones((nROI))
SE[0]=0.001*np.ones((nROI))


#-------------------------------------------------------------- 
# Running the network
#-------------------------------------------------------------- 

for t in range(1,nt):

	longrange_E = np.dot(JN*G*C,SE[t-1])   # Long range excitatory contribution
	
	# Inhibition
	I_inh[t-1] = (wI*I0) + (JN*SE[t-1]) - SI[t-1]
 	#          Back-ground    Excitatory   Inhibitory 
	RI[t-1] = fI(aI,bI,dI,I_inh[t-1])

	# Excitation
	I_exc[t-1] = (wE*I0) + (w*SE[t-1])+ longrange_E - (w_fic*SI[t-1])  # FIC weights here
 	#      Back-ground    Excitatory    Long-range     Inhibitory 
	RE[t-1] = fI(aE,bE,dE,I_exc[t-1])

	
	#status_message("Timept {}".format(round(t_plot[t-1],2)))
	#d={"RE_min": round(min(RE[t-1]),2),
	#"RE_max": round(max(RE[t-1]),2),
	#"RI_min": round(min(RI[t-1]),2),
	#"RI_max": round(max(RI[t-1]),2)}
	#print(d)
	
	# Synaptic gating variables

	d_SE = (-SE[t-1]/tauE) +((1-SE[t-1])*gamma*RE[t-1]/1000)+vE[t-1]  
	d_SI = (-SI[t-1]/tauI) +(RI[t-1]/1000)+ vI[t-1] 

	SE[t] = np.clip(SE[t-1] + (d_SE * dt),0,1)         #First-order Euler method
	SI[t] = np.clip(SI[t-1] + (d_SI * dt),0,1)
	#SE[t] = SE[t-1] + (d_SE * dt)                       #First-order Euler method
	#SI[t] = SI[t-1] + (d_SI * dt)

	# Syaptic variable is clipped by the codes in Dermirtas
	# is it valid analytically?	
	#id={"SE_min": round(min(SE[t]),2),
	#		"SE_max": round(max(SE[t]),2),
	#		"SI_min": round(min(SI[t]),2),
	#		"SI_max": round(max(SI[t]),2)}
	#print(d)

def plot_bold(s,i=(0,1,10,20,30,67),txt="Synaptic gating variable"):
		
		f,ax=plt.subplots(2,3,figsize=(20,5))
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

s=RE  # giving rate as synaptic variable
#s=SE # giving synaptic activity directly

#print(s.shape)
#O=np.ones((2000,nROI))
#Z=np.zeros((nt-2000,nROI))
#s=np.vstack((O,Z))

# Intialization of the variables
nt=s.shape[0]
x=np.zeros((nt,nROI))    # vaso-dilatory signal
f=np.ones((nt,nROI))   # blood inflow
v=np.ones((nt,nROI))    # blood volume
q=np.ones((nt,nROI))    # deoxy hemoglobin 
y=0.03*np.ones((nt,nROI))  # BOLD signal 


#---------------------------------------------------------------------------------
# Running the Ballon model using the synaptic gating variable
#---------------------------------------------------------------------------------
for t in range(1,s.shape[0]):
	#x is Vaso-dilatory time series
	# dx =  s[t-1] - (kappa*x[t-1])- (gamma *(f[t-1]-1))
	dx =  (0.5* s[t-1]) + 3 - (kappa*x[t-1])- (gamma *(f[t-1]-1))   # According to deco
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


#plot_bold(s,txt="Synaptic variables")
#plot_bold(x[:-1000],txt="Vaso-dilatory")
#plot_bold(f[:-1000:],txt="blood in-flow")
#plot_bold(v[:-1000:],txt="blood volume")
#plot_bold(q[:-1000:,],txt="deoxy hemoglobin")
#plot_bold(y[54000:],txt="Percent BOLD change")


ysim=y[54000:]    # Discarding 1 mins of data

print("++ Simulated BOLD time series shape is",ysim.shape)

ysim_m=np.zeros(ysim.shape)
sim_length=ysim.shape

sim_filtered=np.zeros(ysim[::754,].shape)
sim_dsampled_nomean=np.zeros(ysim[::754,].shape)
sim_dsampled=np.zeros(ysim[::754,].shape)  # initializing for downsampling

#--------------------------------------------------------------
# Filtering of fMRI time series data
#--------------------------------------------------------------


fl =0.008           			# lowpass frequency of filter
fh =0.08            			# highpass
k=2                  			# 2nd order butterworth filter
Nq=1/(2*TR)      				# Nyquist frequency
Wn=[fl/Nq,fh/Nq] 				# Normalised frequency
b,a=butter(k,Wn,btype='band')  	# Construct the filter



# removing the mean and filtering as in the Deco
for k in range(nROI):
	# mean centering
	ysim_m[:,k]=ysim[:,k]-ysim[:,k].mean()

	# Downsampled the data before finding functional connectivity
	sim_dsampled_nomean[:,k]=ysim[::754,k]
	sim_filtered[:,k]=filtfilt(b,a,ysim_m[::754,k])
	sim_dsampled[:,k]=ysim_m[::754,k]

print(sim_dsampled_nomean.shape)
print(sim_dsampled.shape)

FCsim=np.zeros((nROI,nROI))
for row in range(nROI):
	for col in range(nROI):
		#temp,_=pearsonr(ysim_m[:,row],ysim_m[:,col])
		temp,_=pearsonr(sim_filtered[:,row],sim_filtered[:,col])
		FCsim[row,col]=temp 
	

#status_message("Printing the simulated functional connectivity")
#print(FCsim)


max_emp=FCemp[~np.eye(*FCemp.shape, dtype=bool)].max()
min_emp=FCemp[~np.eye(*FCemp.shape, dtype=bool)].min()
max_sim=FCsim[~np.eye(*FCsim.shape, dtype=bool)].max()
min_sim=FCsim[~np.eye(*FCsim.shape, dtype=bool)].min()

print("Maximum and minimim values of the fitted data are ", max_sim, min_sim)
print("Maximum and minimim values of the empirical data  are ", max_emp, min_emp)


Emp_up=np.triu(FCemp,1)  # upper off-diagonal elements
Sim_up=np.triu(FCsim,1)
flat_emp= Emp_up.flatten()
flat_sim=Sim_up.flatten()

Emp_upper=flat_emp[flat_emp!=0]
Sim_upper=flat_sim[flat_sim!=0]

print(Emp_upper.shape)
print(Sim_upper.shape)

fit_stat,p_fit =pearsonr(Emp_upper,Sim_upper)  # without Z transform

print("The fit statistics for one simulation is ", fit_stat," with the p-fit of ",p_fit)
plt.scatter(Emp_upper,Sim_upper)
plt.show()


Emp_upper, Sim_upper = np.arctanh(Emp_upper),np.arctanh(Sim_upper)
fit_stat,p_fit =pearsonr(Emp_upper,Sim_upper)

print("The fit statistics for one simulation is ", fit_stat," with the p-fit of ",p_fit)
plt.scatter(Emp_upper,Sim_upper)
plt.show()



"""

fig, (ax1, ax2) = plt.subplots(2,figsize=(3,10))
Simu = ax1.imshow(FCsim, cmap='jet', interpolation='none')

#fig.colorbar(Simu, ax=ax1)

# you can specify location, anchor and shrink the colorbar
Empi = ax2.imshow(FCemp, cmap='jet', interpolation='none')
#fig.colorbar(Empi, ax=ax2)
#cbar.minorticks_on()
plt.show()
 
"""


