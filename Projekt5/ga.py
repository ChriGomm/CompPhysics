from geneticalgorithm import geneticalgorithm as ga
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
dat = []
with open("lichtkurve.txt","r") as file:
    for lines in file:
        dat.append([float(lines.split()[0]),float(lines.split()[1])])

data = np.array(dat)
print(len(data[:,0]))

from copy import deepcopy
from scipy.optimize import minimize
def fit_function(x):
    n = len(x)
    a = x[0]*10
    b= x[1]*100
    amp = x[2::3]*100
    per = x[3::3]*(50-(data[1,0]-data[0,0]))+(data[1,0]-data[0,0])
    phi = x[4::3]*np.pi*2
    fun = 0
    for j in range((n-2)//3):
        fun = deepcopy(fun+amp[j]*np.sin(2*np.pi/per[j]*data[:,0]+phi[j]))
    fun += a*data[:,0]+b
    return np.linalg.norm((fun-data[:,1])**2/25)**2



def iterator(i,a):
    bounds = np.zeros((2+3*i,2))
    bounds[0,:]=np.array([0,20])
    bounds[1,:]=np.array([0,100])
    for k in range(1,1+i):
        bounds[k*3,:]=np.array([(data[1,0]-data[0,0])*2,50])
        bounds[k*3-1,:]=np.array([0,100])
        bounds[1+k*3,:]=np.array([0,2*np.pi])
    bounds[:,0]=np.zeros(i*3+2)
    bounds[:,1]=np.ones(i*3+2)
    algorithm_param = {'max_num_iteration': 300+10000/(1-0.044)*(a[0]-0.02),\
                   'population_size':100,\
                   'mutation_probability':a[0],\
                   'elit_ratio': a[1],\
                   'crossover_probability': a[2],\
                   'parents_portion': max(a[3],a[1]+0.01),\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
    model = ga(fit_function,2+3*i,variable_type='real',variable_boundaries=bounds,algorithm_parameters=algorithm_param)
    model.run()
    gc.collect(2)
    return model
def mean_it(a):
    N =3
    min_val = 10**12
    # parameter = np.zeros(2+n*3)
    for rep in range(7):
        model = iterator(N,a)
        if (model.output_dict['function']<min_val):
            min_val =model.output_dict['function']
            # paramter = model.output_dict['variable']
    return min_val
opt = {"maxiter":100,"disp":True}
a0 = np.array([0.044,0.23,0.9,0.3])
bs = [(0,1),(0,0.9),(0,1),(0,1)]
minimize(mean_it,x0=a0,bounds=bs,options=opt)