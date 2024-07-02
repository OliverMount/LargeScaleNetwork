shift=0
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
