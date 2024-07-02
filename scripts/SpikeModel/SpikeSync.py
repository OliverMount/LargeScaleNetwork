#!/usr/bin/env python

# coding: utf-8

# In[5]:


#import scipy


# In[6]:


#hierVals = scipy.io.loadmat('/home/olive/Desktop/LSN/Jog/hierValspython.mat')
#hierValsnew = hierVals['hierVals'][:]
#hier=hierValsnew/max(hierValsnew)#hierarchy normalized. 
#hier=np.squeeze(hier[:nAreas])


# In[7]:


# Loading the required modules
from __future__ import division
from brian2 import *
prefs.codegen.target = 'auto'

import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import pandas as pd
import numpy.random
import random as pyrand
import time

from brian2 import defaultclock
import brian2genn
#set_device('genn')
#prefs.devices.genn.path='/olive/genn/'
#prefs.devices.genn.cuda_backend.cuda_path="/usr/local/cuda/"
# Parameter settings


## Loading the structural connectivity information
# Hierarchial values
#netwParams_hier = np.load('./netwParams_hiervals.npy')
#distMat = np.load('./distMatval.npy') 
#flnMat = np.load('/home/olive/Desktop/LSN/Jog/flnMatshuf2.npy')         # FLNs
defaultclock.dt=1*ms

dt=defaultclock.dt    
binsize = 5*ms  

block=[10,10,40]*ms  # rest - stimulus - rest
duration=sum(block)

netsteps= int(sum(block/dt))  # Length of the input stimulus signal
 
# reading the distance matrix    
with open("/olive/distMatval.txt") as f:
    contents = f.readlines() 
distMat= np.array([[float(k) for k in i.split()] for i in contents]) 


#plt.imshow(distMat, interpolation='none')
#plt.show()

# reading the FLNmatrix 
#with open("/olive/flnMatshuf2.txt") as f:
#    contents = f.readlines() 
#fln_mat= np.array([[float(k) for k in i.split()] for i in contents]) 

ff=pd.read_csv("/olive/FLNmtx.csv")   
fln_mat=np.asarray(ff)
#plt.imshow(fln_mat, interpolation='none')
#plt.show()


# ROIs
areas =['V1','V2','V4','DP','MT','8m','5','8I','TEO','2','F1','STPc','7A','46d',
        '10','9/46v','9/46d','F5','TEpd','PBr','7m','7B','F2','STPi','PROm','F7',
       '8B','STPr','24c']

# Hierarchial values
hier = np.array([0,0.34,0.4,0.45,0.5,0.55,0.58,0.6,0.61,0.63,0.67,
                 0.7,0.72,0.73,0.76,0.78,0.8,0.83,0.85,0.86,0.87,
                 0.95,0.96,0.965,0.97,0.98,0.985,0.99,1])


#currval,currdur = curr, 1500 
#flnMat = np.load('./flnMatshuf2.npy')
#flnMat = np.tril(flnMat)
 
print("Block duration :", block)
print("Duration of simulation :" ,duration)
 
 


# In[9]:


print(" \n ++ Intializing parameters of the network")
          
        
n = 100
fac=4
Nn=n*fac    

Vr = -70.*mV 
Vreset = -60.*mV
Vt = -50.*mV 
tref = 2.*ms    # Refractory time of a neuron
tauE = 20. * ms # Tau of excitatory neuron  
tauI= 10. * ms  # Tau of inhibitory neuron 

plocal = 0.1 # probability of neural connections (local only)
plongr= 0.1    # probability of long range connectivities 
isFB = True 
A=10000               # Input signal amplitude
sig =  3*mV       # % of signal is noise here   
dlocal = 0      # delay(for within area) 
speed = 3.5
  
muI  = 14.7*mV 
muE = 14.2*mV

muEE  = .27*mV
muIE = .19*mV

wEE = .01*mV
wEI = .98*mV 

wII = .3*mV
wIE =.3*mV

eta  = 4
""" 
fln_mat = np.array([[0,0.127,0.0005,0.0001], 
          [0.127,0,0.0005,0.001], 
          [0.0005,0.0005,0,0.0023],
          [0.0001,0.0001,0.0023,0]])

hier = np.array([0,0.25,0.6,0.95])
areas =['V1','V4','7A','9/46d']  
distMat=np.array([[0,14.8,25.5,55],
                  [14.8,0,22.8,45.9],
                  [25.5,22.8,0,35.5],
                  [55,45.9,35.5,0]])
"""

                  
delayMat = distMat/speed
arealen = len(areas) 

print(" \n ++ Done with initialization\n")

print(" ++ Duration of the network run would be {} sec \n".format(duration))


# #### Defining the input ( random for neuron sounding and pulse for response)

# In[ ]:





""" #try input to area 2
currvalarea2 = 15.8
a2area2 = currvalarea2*np.ones([currdur,1])
aareaarea2 = np.vstack((a1, a2area2, a3))
timelen = len(aareaone)
excotherareasone, excotherareastwo = para['k']*4*8, para['k']*4*19
aareaonenet, aareaarea2net = np.tile(aareaone,(1,para['k']*4)), np.tile(aareaarea2,(1,para['k']*4))
arestone, aresttwo = np.zeros([timelen, excotherareasone]), np.zeros([timelen, excotherareastwo])
netarr = np.hstack((aareaonenet,arestone,aareaarea2net,aresttwo)) #new for fig. 
"""

print("\n Creating input waveforms for all neurons and areas...")

signal= concatenate((zeros(round(block[0]/dt)),
                 A*ones(round(block[1]/dt)),
                 zeros(round(block[2]/dt))))
signal=signal[:,np.newaxis]

timelen = len(signal)
N_otherareas =  n*4*(arealen-1)
aareaonenet = np.tile(signal,(1,n*4))
arest = np.zeros([timelen, N_otherareas])
netarr = np.hstack((aareaonenet,arest))


Input = TimedArray(netarr*mV, dt=dt)

print("\n Done with input waveform creation...\n")

# # Network setup
# 1. Setting spike equation
# 2. Neuron groups
# 3. Defining synapses and connectivity
# 4. Running the network
# 5. Recording the spikes and rates
# 

# #### 1. Setting spike equation 
# 
# 
# 
# 

# In[ ]:


# For excitatory neurons

eqsE = Equations(''' dV/dt= (-(V-Vr) + muE+Input(t,i))*(1./tauE) + (sig*(1./tauE)**0.5)*xi : volt (unless refractory) ''' ) 

#eqsE = Equations(''' dV/dt=(-(V-Vr) + muE+Input(t,i))*(1./tauE) : volt (unless refractory) ''' ) 
# For inhibitory neurons

eqsI = Equations('''
  dV/dt=(-(V-Vr) + muI )*(1./tauI) + (sig*(1./tauI)**0.5)*xi : volt (unless refractory)
  ''')


# #### 2. Making Neuron group

# In[7]:


E = NeuronGroup(Nn*arealen,    # No of  Excitatory  neurons = (No of Inhibitory/4) 
                method='euler', 
                model=eqsE, 
                threshold='V > Vt', 
                reset='V=Vreset', 
                refractory='tref')

I = NeuronGroup(n*arealen,
                method='euler',
                model=eqsI, 
                threshold='V > Vt', 
                reset='V= Vreset',
                refractory='tref')


# In[8]:


# Assigning Neurons to each areas

Exc = [ E[y*Nn:((y+1)*Nn)] for y in range(arealen)]
Inh = [ I[z*n:(z+1)*n] for z in range(arealen)] 


# In[9]:


print("No. of excitatory neurons in each ROI is ",len(Exc[1]))
print("No. of inhibitory neurons in each ROI is ",len(Inh[1]))




# #### 3. Defining synapses (within area  and between area synaptic connectivity)

# In[10]:

start_time=time.time()
print("\n Defining synapses and connectivity ... ")
Exc_C_loc, Inh_C_loc, EtoI_C_loc, ItoE_C_loc = [None]*arealen, [None]*arealen, [None]*arealen, [None]*arealen 
Exc_C_lr_fromi, EtoI_C_lr_fromi =[], []

for h in range(arealen):
    
    ############ LOCAL ##################
    
    print("Defining recurrent synapse for area : ", h)
    # Recurrent excitation and inhibition
    Exc_C_loc[h] = Synapses(Exc[h], Exc[h], 'w:volt', delay = dlocal*ms, on_pre='V+=w')  
    Inh_C_loc[h] = Synapses(Inh[h], Inh[h], 'w:volt', delay = dlocal*ms, on_pre='V+= w ')  
    
    print("Defining recurrent connections for area : ", h)
    Exc_C_loc[h].connect(p = plocal) #this step is taking longest time. rate determining. 
    Inh_C_loc[h].connect(p = plocal) 
    
    Exc_C_loc[h].w = (1+eta*hier[h])*wEE
    Inh_C_loc[h].w = -wII
    
    print("Defining feedback synapse for area : ", h)
    # feeback excitation and inhibition
    EtoI_C_loc[h] = Synapses(Exc[h], Inh[h], 'w:volt', delay = dlocal*ms, on_pre='V+= w ')    
    ItoE_C_loc[h] = Synapses(Inh[h], Exc[h], 'w:volt', delay = dlocal*ms, on_pre='V+= w ') 

    
    print("Defining feedback synapse for area : ", h)
    EtoI_C_loc[h].connect(p = plocal) 
    ItoE_C_loc[h].connect(p = plocal) 

    
    EtoI_C_loc[h].w = (1+eta*hier[h])*wIE
    ItoE_C_loc[h].w = -wEI

    ############ GLOBAL, long-range connections ##################  
    for m in range(arealen):
      if m!= h:  
          exc_lr_itoj, etoi_lr_itoj = None, None
 
          exc_lr_itoj = Synapses(Exc[h], Exc[m], 'w:volt', on_pre='V+= w ') 
          etoi_lr_itoj = Synapses(Exc[h], Inh[m], 'w:volt', on_pre='V+= w ')

          exc_lr_itoj.connect(p = plongr)  #long time.   
          etoi_lr_itoj.connect(p = plongr)  

          exc_lr_itoj.w =  (1 + eta * hier[m]) * muEE * fln_mat[m,h]
          etoi_lr_itoj.w = (1 + eta * hier[m]) * muIE * fln_mat[m,h]

          #meanlr, varlr = delayMat[m,h], .1*delayMat[m,h]
          #exc_lr_itoj.delay = np.random.normal(meanlr,varlr,len(exc_lr_itoj.w))*ms
          #etoi_lr_itoj.delay = np.random.normal(meanlr,varlr,len(etoi_lr_itoj.w))*ms

          Exc_C_lr_fromi.append(exc_lr_itoj)
          EtoI_C_lr_fromi.append(etoi_lr_itoj) 

print("\n Done with synaptic connections...")
print("\n---%s seconds---\n"%(time.time()-start_time))
# #### 4. Running the network

# In[11]:


# Spike monitoring
M= SpikeMonitor(E) 

# State monitoring
# S = StateMonitor(E, 'V', record=True)


# In[12]:


# Setup the network, and run it
E.V =  Vr  + rand(len(E)) * (Vt - Vr)
I.V =  Vr  + rand(len(I)) * (Vt - Vr)

print("++ Creating the network")
net = Network(E,I,Exc_C_loc,EtoI_C_loc,
              ItoE_C_loc,Inh_C_loc,
              Exc_C_lr_fromi,EtoI_C_lr_fromi,
              M)
#net=Network(M)
print("++ Done with creating the network")
print("++ Running the network \n")

#net.run(duration, report='text',profile=True)

net.run(duration)

#run(duration,report='text')
# net.profiling_info
#run(duration,report='text')
print("\n++ Done with the network run")

print("\n---%s seconds---\n"%(time.time()-start_time))

# #### 5. Recording spikes and rates

# ##### Spike sorting

# In[13]:


# Variable M contains the spike information

netspike = len(M.i)
allspike = np.empty([netspike,2])
allspike[:,0]=M.t/ms               # events
allspike[:,1]=M.i                # Neuron number
allspikesorted = allspike[allspike[:,1].argsort(),]

allspikesorted  # 2D array of spike times on 1st column and 
                # neuron number on the 2nd column


print(allspikesorted.shape)
# In[14]:


# The variable 'allspikesorted' is exploited here to arrange it in the list of area spikes
# So Spk is the Spike information in the list with each area is an element of the list
# An area contains Nn number of neurons
 
Spk=[]  # This is list of 2D array. Each 2D array is for an area that contains spikes

for k in range(arealen): 
    
    if not k:
        Ind1=np.where(allspikesorted[:,1]<(Nn-1)*(k+1)+k)[-1][-1]
        Spk.append(allspikesorted[0:Ind1,:])
    else:
        Ind2=np.where(allspikesorted[:,1]<(Nn-1)*(k+1)+k)[-1][-1] 
        Spk.append(allspikesorted[(Ind1+1):Ind2,:])
        Ind1=Ind2 

# In[15]:


len(Spk)


# In[16]:

"""
## Raster for different areas

figure(figsize=(14,4))
 
#for k in range(arealen): 
for k in range(3): 
    plt.subplot(int(str(41) +str(k+1)))   
    plot(Spk[k][:,0],Spk[k][:,1],'.b')#-(k*(Nn-1))
    plt.title(areas[k])
    plt.ylabel("No. Neurons")
    plt.xlabel('Time (msec)')
    #plt.axvspan(block[0]/ms, block[0]/ms + block[1]/ms, alpha=0.5, color='green')
     
plt.show() 

# In[17]:
"""

block


# ##### 6. Obtain rates from the spikes

# In[18]:


maxrate = np.empty([arealen,1])     # max rate
meanrate = np.empty([arealen,1])    # mean rate

netbinno = int( 1+(duration/ms)-(binsize/ms))
poprate = np.empty([arealen,netbinno ])   #pop-rate (instantaneous rate)


# In[19]:


poprate


# In[20]:


Nn


# In[21]:


count = 0               #for each spike. 
stepsize = 1*ms
monareaktimeall = []
for u in range(arealen):  # For each ROI
      monareaktime = []

      while((count < netspike) and (allspikesorted[count,1]<(Nn)*(u+1)) ):
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

      valsrate = valsnew*((1000*ms/binsize) /(Nn) ) #new divide by no of neurons per E pop. 
      poprate[u,:] = valsrate   
      maxrate[u,0] = max(valsrate[int(len(valsrate)/3):])
      monareaktimeall.append(monareaktime)
    


# In[22]:

"""
figure(figsize=(14,8))
shift=10
for q in range(arealen): 
 
    plt.subplot(int(str(22) +str(q+1)))   
    plt.plot(poprate[q,shift:])
    plt.title(areas[q])
    plt.ylabel("Firing rate (Hz)")
    plt.xlabel('Time (msec)')
    #plt.plot(range(3))
    plt.axvspan((block[0]/ms)-shift, (block[0]/ms) + (block[1]/ms)-shift, alpha=0.1, color='green')
    plt.axis([0, 1000, min(poprate[q,:]), max(poprate[q,:])])
     
plt.show()    
     
"""
# saving the population rates and spike data as numpy files

np.save('poprate.npy',poprate)
np.save('spikes.npy',np.asarray(Spk),allow_pickle=True)

# In[23]:



max_rate=[(areas[k], max(poprate[k,:])) for k in range(arealen)]
max_rate

df= pd.DataFrame(max_rate)

df


# In[27]:


#[(areas[k], np.argmax(poprate[k,:])) for k in range(arealen)]


# In[28]:


#dt


# In[29]:


#poprate.shape


#_=plt.figure(figsize=(4,4))
#area_name_list=areas
#area_idx_list =


shift =0
T=sum(block)
t_plot = np.linspace(0, T, int(T/dt)+1)

_ = plt.figure(figsize=(14,4))
area_name_list  = areas
area_idx_list   = [-1]+[areas.index(name) for name in area_name_list]
f, ax_list      = plt.subplots(len(area_idx_list), sharex=True)

for ax, area_idx in zip(ax_list, area_idx_list):
    if area_idx < 0:
        y_plot  = signal 
        txt     = 'Input'
    else:
        y_plot  = poprate[area_idx,shift:]
        txt     = areas[area_idx]

    y_plot = y_plot - y_plot.min()
    tplot=t_plot[0:len(y_plot)]
    ax.plot(tplot*1000, y_plot)
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



## plotting spikes

area_idx_list   = [areas.index(name) for name in area_name_list]
f, ax_list      = plt.subplots(len(area_idx_list), sharex=True)

for ax, area_idx in zip(ax_list, area_idx_list):
    y_plot  = Spk[area_idx][:,1]
    txt     = areas[area_idx]

    y_plot = y_plot - y_plot.min()
    ax.plot(Spk[area_idx][:,0],y_plot,'.b')  #-(k*(Nn-1))
    ax.text(0.9, 0.6, txt, transform=ax.transAxes)

    ax.set_yticks([y_plot.max()])
    ax.set_yticklabels(['{:0.4f}'.format(y_plot.max())])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

f.text(0.01, 0.5, 'Neurons', va='center', rotation='vertical')
ax.set_xlabel('Time (ms)')


plt.show() 
