import numpy as np
import pandas as pd

with open('analytical shocktube solution.txt','r') as file:
    lines = file.readlines()
data =[]
for line in lines[2:]:
    data.append(list(map(float,line.split())))
   
data = np.array(data)



import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)
ax.plot(data[:,1],data[:,3])
ax.set_xlabel('x-Position') 
ax.set_ylabel(r'$\rho(x)$')


fig2, ax2 = plt.subplots(1,1)
ax2.plot(data[:,1],data[:,2])
ax2.set_xlabel('x-Position') 
ax2.set_ylabel(r'$u(x)$')

fig3, ax3 = plt.subplots(1,1)
ax3.plot(data[:,1],data[:,4])  
ax3.set_xlabel('x-Position') 
ax3.set_ylabel(r'$T(x)$')

fig4, ax4= plt.subplots(1,1)
ax4.plot(data[5:-5,1],data[5:-5,5]) 
ax4.set_xlabel('x-Position') 
ax4.set_ylabel(r'$p(x)$')
plt.show()