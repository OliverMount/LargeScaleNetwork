It=mat73.loadmat(full_path_e)
IE=It['xn'].T

print("Shape of the current time series is ",IE.shape)
#It=mat73.loadmat(full_path_i)
#II=It['xg'].T
#II=II[:5000,:]


#print(II.shape)

sig_len= 60000  # in msec

shift=5           # Time series shift for avoiding initial spikes
N= sig_len + shift  # Total number of time points in the input current
IE=IE[:N,:]

Nroi=10

True_another=np.zeros((N,Nroi))

for l in range(Nroi):
    True_another[:,l]=IFcurve(aE,bE,dE,IE[:,l]*namp)


print("++ Plotting the rates \n")
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
    ax.set_ylim(0,20)
    if area_idx==0:	
        ax.legend()

f.text(0.09, 0.5, 'Firing rate (Hz)',  rotation='vertical',size=14)
f.set_size_inches(18.5, 10.5)
ax.set_xlabel('Time (ms)',size=16)
ax.set_xlim(1,sig_len)
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
    
    #ax.legend()

f.text(0.09, 0.5, 'Neuron Index', va='center', rotation='vertical',size=14)
f.set_size_inches(18.5, 10.5)
ax.set_xlabel('Time (ms)',size=16)
ax.set_xlim(1,sig_len)
show()
