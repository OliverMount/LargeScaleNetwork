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





