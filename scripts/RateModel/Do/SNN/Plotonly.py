figure(figsize=(18,4))
plt.plot(poprate[22,1000:9000],'b')
plt.plot(True_rate[1000:9000,22],'r')
plt.show() 
