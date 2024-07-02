start_scope()
defaultclock.dt  = 1*ms
dt=defaultclock.dt

run_time=N
dlocal=0*ms 
 
(R, Vrest,Vreset,Vt,tref,tauE,sig,dt)= NeuronParameters(R=43*Mohm,sig=2*mvolt)
 

## Forming the input matrix for each ROI
I_exci=np.zeros((N,N_e*N_ROI)) 
I_inhi=np.zeros((N,N_i*N_ROI)) 

for l in range(N_ROI):
    I_exci[:,l*N_e:((l+1)*N_e)]=A[:,:N_e,l] 
    I_inhi[:,l*N_i:((l+1)*N_i)]=A[:,N_e:,l] 
 
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

print("++ Establishing synaptic connections")

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
    WII = [-item for sublist in Wroi[h + (h*(3))+1] for item in list(sublist)]
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
    WEI = [-item for sublist in Wroi[h + (h*(3))+3] for item in list(sublist)]
    IE_loc[h].w = WEI  

    # For each local pool there is a long range connection

    for m in range(N_ROI-1):
      if m!= h:  
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

print("++ Done with establishing connection \n")
print("++ Running the network \n")

#M=StateMonitor(E,'V',record=True) 
S=SpikeMonitor(E)
run(N*ms)

print("++ Calculating rate from the network \n")

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
area_name_list  = ROI_names[:5]
area_idx_list   = [ROI_names.index(name) for name in area_name_list]
f, ax_list      = plt.subplots(len(area_idx_list), sharex=True)
 
for ax, area_idx in zip(ax_list, area_idx_list):
  
    y_plot  = poprate[area_idx,:] 
    txt     = ROI_names[area_idx]

    y_plot = y_plot - y_plot.min()
    ax.plot(y_plot,"r",label="Spike Model")
    ax.plot(True_rate[:,area_idx],"b",label="Rate-model")
    ax.text(0.8, 0.8, txt, transform=ax.transAxes,size=16)

    
    #ax.set_yticks([y_plot.max()])
    #ax.set_yticklabels(['{:0.1f}'.format(y_plot.max())])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim(1,5000)
    ax.set_ylim(0,10)
    ax.legend()

f.text(0, 0.5, 'Firing rate (Hz)',  rotation='vertical',size=16)
f.set_size_inches(18.5, 10.5)
ax.set_xlabel('Time (ms)',size=16)
show()
