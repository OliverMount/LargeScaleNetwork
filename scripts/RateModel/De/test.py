# Testing the solution of differential equation

import numpy as np
import matplotlib.pyplot as plt

dt=0.1
time= np.arange(0,10,dt)
n=len(time)

y=np.zeros((n))
y[0]=1  # initial value

for t in range(1,n):
	dy= -y[t-1]
	y[t] = y[t-1] + dy*dt

plt.plot(time,y,'k',label="Numerical")
plt.plot(time,np.exp(-time),'b--',label="Analytical")
plt.legend()
plt.show()

	
