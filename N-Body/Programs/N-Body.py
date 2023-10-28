 
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.constants as cons
import numpy as np
import pandas as pd
import pathlib
 
def kl(wert_a,wert_b):
    if wert_a < wert_b:
        return True
    else:
        return False
 
def fac(i):
    result = 1
    for x in range(1,i+1):
        result = result*x
    return result
 
 
with open("in2.txt", "r") as datafile:
    readfile = csv.reader(datafile)
    data = []
    for row in readfile:
        str_row = row[0].split()
        int_row = []
        for i in str_row:
            int_row.append(float(i))
        data.append(int_row)
   
    N = int(data[0][0])
    t_max = data[0][1]
    t_delta = data[0][2]/100
    m= np.array(data[1:N+1])
    r = np.array(data[N+1:2*N+1])
    v = np.array(data[2*N+1:])


     
 
 
# schwerpunkt trafo
R = sum(m*r)/sum(m)
RV = sum(m*v)/sum(m)
r_transformed = r-R
v_transformed = v-RV
m_trans = m/sum(m)
 
 
 
 
zero = np.zeros((N,3))
 
def a_vec2(pos):
    dist = np.zeros((N, N, 3))
    for i in range(3):
        dist[:,:,i] = -np.subtract.outer(pos[:, i], pos[:, i])
    #dist = np.subtract.outer(r, r) # not yet -> (N x N x 3)
    betrag1 = np.linalg.norm(dist, axis = 2)
    betrag = np.copy(betrag1)
    np.place(betrag,betrag==0,[1])
    fraction= np.zeros((N, N, 3))
    for i in range(3):
        fraction[:,:,i] = dist[:,:,i] / (betrag * betrag * betrag)
   
    return np.dot(m[:,0], fraction), betrag1 # (N x 3)
 
# dist = np.zeros((N,N,3))
# for n in range(N):
#     for j in range(N):
#         if n!=j:
#             dist[n,j,:]= (r_transformed[n,:]-r_transformed[j,:])/np.dot(r_transformed[n,:]-r_transformed[j,:],r_transformed[n,:]-r_transformed[j,:])**(3/2)
#         else:
#             for i in range(3):
#                 dist[n,n,i]= 0
 
# dist2 = np.zeros((N,N,3))
# for i in range(3):
#         dist2[:,:,i] = np.subtract.outer(r_transformed[:, i], r_transformed[:, i])
# betrag = np.linalg.norm(dist2, axis = 2)
# fraction= np.zeros((N, N, 3))
# for i in range(3):
#     fraction[:,:,i] = dist2[:,:,i] / (betrag * betrag * betrag)
#     np.nan_to_num(fraction,copy=False)
# print(dist==fraction)
# print(dist)
# print(fraction)
 
def a_vec(pos):
    a = np.zeros((N,3))
    for i in range(0,N):
        store = np.zeros((N,3))
        for j in range(0,N):
            if i!=j:
                store[j,:] = (m[j,0]*(pos[j,:]-pos[i,:])/(np.dot(pos[j,:]-pos[i,:],pos[j,:]-pos[i,:])**(3/2)))
        a[i,:] =sum(store)
    return a ,pos
 
# print(a_vec(r_transformed)[0]==a_vec1(r_transformed))
 
def dt_a(pos,vel):
    a = np.zeros((N,3))
    for i in range(0,N):
        store = np.zeros((N,3))
        for j in range(0,N):
            if i!=j:
                store[j,:] = (m[j,0]*((vel[j,:]-vel[i,:])/(np.dot(pos[j,:]-pos[i,:],pos[j,:]-pos[i,:])**(3/2))-3*np.dot(vel[j,:]-vel[i,:],pos[j,:]-pos[i,:])*(pos[j,:]-pos[i,:])/(np.dot(pos[j,:]-pos[i,:],pos[j,:]-pos[i,:])**(5/2))))
        a[i,:] = sum(store)
    return np.array(a)
 
def dt_a2(pos,vel):
    dist = np.zeros((N, N, 3))
    distv = np.zeros((N, N, 3))
    for i in range(3):
        dist[:,:,i] = -np.subtract.outer(pos[:, i], pos[:, i])
    for i in range(3):
        distv[:,:,i] = -np.subtract.outer(vel[:, i], vel[:, i])
    #dist = np.subtract.outer(r, r) # not yet -> (N x N x 3)
    betrag = np.linalg.norm(dist, axis = 2)
    np.place(betrag,betrag==0,[1])
    fractionv= np.zeros((N, N, 3))
    for i in range(3):
        fractionv[:,:,i] = distv[:,:,i] / (betrag * betrag * betrag)
    v_dot_r = np.einsum('ijij->ij',np.tensordot(fractionv,dist,(2,2)))/betrag/betrag
    part2 = np.zeros((N,N,3))
    for i in range(3):
        part2[:,:,i] = dist[:,:,i]*v_dot_r
    whole = fractionv-3*part2
   
 
    return np.dot(m[:,0],whole) # (N x 3)
 
# def func(a,b):
#     return a/b
# rez = np.vectorize(func)
 
def energy1(pos,vel,betrag):
    v_sq= np.sum(vel*vel,axis=1)
    T= np.dot(m[:,0],v_sq)/2
    V= np.zeros(N)
    for n in range(N):
        norm = np.linalg.norm(pos-pos[n],axis=1)
        np.place(norm,norm==0,['inf'])
        V[n]= np.dot(m[:,0],m[n,0]*1/norm)/2
    V = sum(V)
    return T-V
 
def energy(pos,vel,betrag):
    T = np.zeros(N)
    for n in range(N):
        T[n]= m[n,0]*np.dot(vel[n,:],vel[n,:])/2
    T = sum(T)
    V = np.zeros(N)
    for i in range(0,N):
        store = np.zeros((N))
        for j in range(0,N):
            if i!=j:
                store[j] = (m[j,0]*m[i,0]/(np.dot(pos[j,:]-pos[i,:],pos[j,:]-pos[i,:])**(1/2)))
        V[i] =sum(store)
    V = sum(V)/2
    return T-V
   
        
    
 
def energy2(pos,vel, betrag):
    T = np.dot(m[:,0],np.einsum('ii->i',np.tensordot(vel,vel,(1,1))/2))
    np.place(betrag,betrag==0,['inf'])
    r_rez = 1/betrag
  
    V = np.dot(m[:,0],np.dot(m[:,0],r_rez))/2
    return T-V
 
def total_angular_momentum(pos,vel):
   
    result = np.dot(m[:,0],np.cross(pos,vel,axisa=1,axisb=1))
    return result
 
def total_momentum(vel):
    result = np.dot(m[:,0],vel)
    return result
 
 
# cross = np.zeros((N,3))
# for n in range(N):
#     cross[n,:] = np.cross(r_transformed[n,:],v_transformed[n,:])
# print(cross==np.cross(r_transformed,v_transformed,axisa=1,axisb=1))
   
#für das Zweikörper für Aufgabe 2 Problem(noch nicht implementiert):
 
def j_vec(pos,vel):
    r = pos[0,:]-pos[1,:]
    v = vel[0,:]-vel[1,:]
    return r, v, np.cross(r,v)
 
def runge_lenz(r,v,j):
    return np.cross(v,j)/sum(m)-r/np.linalg.norm(r)
 
 
# iteratoren


 
 
def euler(pos,vel,tdelt):
    a,rij_betrag = a_vec(pos)
    dta = dt_a(pos,vel)
    vel1 = vel+a*tdelt
    pos = pos+vel*tdelt
    return pos, vel1, a, dta, rij_betrag
   
def euler_cromer(pos,vel,tdelt):
    a,rij_betrag = a_vec(pos)
    dta = dt_a(pos,vel)
    vel= vel+a*tdelt
    pos = pos+ vel*tdelt
    return pos, vel, a, dta, rij_betrag
 
def velo_verlet(pos,vel,a1,tdelt):
    
    pos = pos + vel*tdelt+a1*tdelt**2/2
    a2,rij_betrag = a_vec(pos)
    vel = vel +(a1+a2)*tdelt/2
    return pos, vel, a2, rij_betrag
 
def kick_drift(pos,vel,a1,tdelt):
    dta = dt_a(pos,vel)
    vel1 = vel+a1*tdelt/2
    pos = pos +vel1*tdelt
    a2,rij_betrag = a_vec(pos)
    vel = vel1 + a2*tdelt/2
    return pos, vel, a2, dta, rij_betrag
 
# Interpolationsfunktion benutzt in Hermite-Iteratoren
 
def interp(*Ord,tdelt):
    store = np.empty((len(Ord),N,3))
    for i in range(0,len(Ord)):
        store[i,:,:]= Ord[i]*tdelt**i/fac(i)
    return sum(store)
 
 
# print(interp(r_transformed,v_transformed,a_vec(r_transformed),dt_a(r_transformed,v_transformed),tdelt=t_delta)==r_transformed+v_transformed*t_delta+a_vec(r_transformed)*t_delta**2/2+dt_a(r_transformed,v_transformed)*t_delta**3/6)
def hermite(pos,vel,tdelt):
    a,rij_betrag = a_vec(pos)
    dta = dt_a(pos,vel)
    vp = interp(vel,a,dta,tdelt=tdelt)
    rp = interp(pos,vel,a,dta,tdelt=tdelt)
    ap,waste = a_vec(rp)
    dtap = dt_a(rp,vp)
    dt2a = -6*(a-ap)/tdelt**2-2*(2*dta+dtap)/tdelt
    dt3a = 12*(a-ap)/tdelt**3+6*(dta+dtap)/tdelt**2
    rc = interp(rp,zero,zero,zero,dt2a,dt3a,tdelt=tdelt)
    vc = interp(vp,zero,zero,dt2a,dt3a,tdelt=tdelt)
    return rc, vc, a, dta, dt2a, dt3a, rij_betrag
 
def hermite_iterator(pos,vel,a,ap,dta,dtap):
    vc = vel + (a+ap)*delta_t/2+(dtap-dta)*delta_t**2/12
    rc = pos + (vc+vel)*delta_t/2+(ap-a)*delta_t**2/12
    ac,rij_betrag = a_vec(rc)
    dtac = dt_a(rc,vc)
    return rc, vc, ac, dtac, rij_betrag
 
def iterated_hermite(pos,vel,a, dta,tdelt):
    vp = interp(vel,a,dta,tdelt=tdelt)
    rp = interp(pos,vel,a,dta,tdelt=tdelt)
    ap, waste = a_vec(rp)
    dtap = dt_a(rp,vp)
    rc = pos
    vc = vel
    dt2a = -6*(a-ap)/tdelt**2-2*(2*dta+dtap)/tdelt
    dt3a = 12*(a-ap)/tdelt**3+6*(dta+dtap)/tdelt**2
    for i  in range(0,2):
        rc, vc, ap, dtap, rij_betrag = hermite_iterator(rc,vc,a,ap,dta,dtap)
    return rc, vc, ap, dtap, dt2a, dt3a, rij_betrag
 
def heun(pos,vel,tdelt):
    a, rij_betrag = a_vec(pos)
    dta = dt_a(pos,vel)
    v1 = a*tdelt
    r1 = vel*tdelt
    v2 = a_vec(pos+r1)[0]*tdelt
    r2 = (vel+v2)*tdelt
    v_iter=vel+(v1+v2)/2
    r_iter = pos +(r1+r2)/2
    return r_iter,v_iter,a, dta, rij_betrag
 
def rk4(pos,vel,tdelt):
    a, rij_betrag = a_vec(pos)
    dta = dt_a(pos,vel)
    v1 = a*tdelt
    r1= vel*tdelt
    v2  = a_vec(pos+r1/2)[0]*tdelt
    r2 = (vel+v1/2)*tdelt
    v3 = a_vec(pos+r2/2)[0]*tdelt
    r3 = (vel+v2/2)*tdelt
    v4 = a_vec(pos+r3)[0]*tdelt
    r4 = (vel+v3)*tdelt
    r_iter = pos +(r1+r4)/6+(r2+r3)/3
    v_iter = vel + (v1+v4)/6+(v2+v3)/3
    return r_iter, v_iter, a, dta, rij_betrag
 
 
# absolute values
def absol(x):
    absolute = []
    for vec in x:
        absolute.append(np.dot(vec,vec)**(1/2))
    return np.array(absolute)
 
def time_delta(a_iter,dta_iter):
    np.place(dta_iter,dta_iter==0,[0.0001])
   
    
    return min(absol(a_iter)/absol(dta_iter))
 
 
def time_delta_hermite(a,dta,dt2a,dt3a):
   return min((absol(a)*absol(dt2a)+absol(dta)**2)/(absol(dta)*absol(dt3a)+absol(dt2a)**2))**(1/2)*t_delta
 
# hier habe ich verschiedene Fälle unterschieden um die Schleifen an die
#  unterschiedlichen iteratoren anzupassen
iterator = hermite




if iterator==hermite or iterator==iterated_hermite:
    t_sum = 0
    delta_t=t_delta
    t_max = 62
    r_iter= r_transformed
    v_iter= v_transformed
    steps = 120000*45
    course_r = np.zeros((steps//15,N,3))
    course_v = np.zeros((steps//15,N,3))
    course_t = np.zeros((steps//15))
    course_r[0,:,:]=r_iter
    course_v[0,:,:]=v_iter
    course_t[0]= 0
    c_energy = np.zeros((steps//15))
    c_totm = np.zeros((steps//15,3))
    c_tot_angm = np.zeros((steps//15,3))
    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
    c_totm[0,:] = total_momentum(v_iter)
    a,rij_betrag = a_vec(r_iter)
    c_energy[0]= energy(r_iter,v_iter,rij_betrag)
    dta = dt_a(r_iter,v_iter)
    print(int(t_max/t_delta*5))
    for zähler in range(0,steps-14):
        t_sum= t_sum+delta_t
        if zähler%15==0:
            course_t[zähler//15] =t_sum
        r_iter,v_iter,a,dta,dt2a,dt3a, rij_betrag = iterator(r_iter,v_iter,delta_t)
        delta_t = time_delta_hermite(a,dta,dt2a,dt3a)
        if kl(delta_t,abs(t_delta-0.3*t_delta))==True:
            delta_t=abs(t_delta-0.3*t_delta)
        elif kl(t_delta+t_delta,delta_t)==True:
            delta_t=t_delta+t_delta
        if zähler%15==0:
            
            c_energy[zähler//15]= energy(r_iter,v_iter,rij_betrag)
            c_tot_angm[zähler//15,:] = total_angular_momentum(r_iter,v_iter)
            c_totm[zähler//15,:] = total_momentum(v_iter)
            course_r[zähler//15,:,:] =r_iter
            course_v[zähler//15,:,:] =v_iter
        
        if zähler%100000 == 0:
            print(zähler)

elif iterator==iterated_hermite:
    t_sum = 0
    delta_t=t_delta
    t_max = 62
    r_iter= r_transformed
    v_iter= v_transformed
    steps = 120000*45
    course_r = np.zeros((steps//15,N,3))
    course_v = np.zeros((steps//15,N,3))
    course_t = np.zeros((steps//15))
    course_r[0,:,:]=r_iter
    course_v[0,:,:]=v_iter
    course_t[0]= 0
    c_energy = np.zeros((steps//15))
    c_totm = np.zeros((steps//15,3))
    c_tot_angm = np.zeros((steps//15,3))
    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
    c_totm[0,:] = total_momentum(v_iter)
    a,rij_betrag = a_vec(r_iter)
    c_energy[0]= energy(r_iter,v_iter,rij_betrag)
    dta = dt_a(r_iter,v_iter)
    print(int(t_max/t_delta*5))
    for zähler in range(0,steps-14):
        t_sum= t_sum+delta_t
        if zähler%15==0:
            course_t[zähler//15] =t_sum
        r_iter,v_iter,a,dta,dt2a,dt3a, rij_betrag = iterator(r_iter,v_iter,a,dta,delta_t)
        delta_t = time_delta_hermite(a,dta,dt2a,dt3a)
        if kl(delta_t,abs(t_delta-0.3*t_delta))==True:
            delta_t=abs(t_delta-0.3*t_delta)
        elif kl(t_delta+t_delta,delta_t)==True:
            delta_t=t_delta+t_delta
        if zähler%15==0:
            
            c_energy[zähler//15]= energy(r_iter,v_iter,rij_betrag)
            c_tot_angm[zähler//15,:] = total_angular_momentum(r_iter,v_iter)
            c_totm[zähler//15,:] = total_momentum(v_iter)
            course_r[zähler//15,:,:] =r_iter
            course_v[zähler//15,:,:] =v_iter
        
        if zähler%100000 == 0:
            print(zähler)
elif iterator == velo_verlet:
    t_sum = 0
    delta_t=t_delta
    t_max = 62
    r_iter= r_transformed
    v_iter= v_transformed
    steps = 3000000
    course_r = np.zeros((steps//15,N,3))
    course_v = np.zeros((steps//15,N,3))
    course_t = np.zeros((steps//15))
    course_r[0,:,:]=r_iter
    course_v[0,:,:]=v_iter
    course_t[0]= 0
    c_energy = np.zeros((steps//15))
    c_totm = np.zeros((steps//15,3))
    c_tot_angm = np.zeros((steps//15,3))
    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
    c_totm[0,:] = total_momentum(v_iter)
    a_iter,rij_betrag = a_vec(r_iter)
    c_energy[0]= energy(r_iter,v_iter,rij_betrag)
    for zähler in range(0,steps):
       
        t_sum= t_sum+delta_t
        if zähler%15==0:
            course_t[zähler//15] =t_sum
        r_iter, v_iter, a_iter,rij_betrag = iterator(r_iter,v_iter,a_iter,delta_t)
       
        if zähler%15==0:
            
            c_energy[zähler//15]= energy(r_iter,v_iter,rij_betrag)
            c_tot_angm[zähler//15,:] = total_angular_momentum(r_iter,v_iter)
            c_totm[zähler//15,:] = total_momentum(v_iter)
            course_r[zähler//15,:,:] =r_iter
            course_v[zähler//15,:,:] =v_iter
        if zähler%50000==0:
            print(zähler)


elif iterator == kick_drift:
    t_sum = 0
    delta_t=t_delta
    t_max = 62
    r_iter= r_transformed
    v_iter= v_transformed
    steps = 300000
    course_r = np.zeros((steps//15,N,3))
    course_v = np.zeros((steps//15,N,3))
    course_t = np.zeros((steps//15))
    course_r[0,:,:]=r_iter
    course_v[0,:,:]=v_iter
    course_t[0]= 0
    c_energy = np.zeros((steps//15))
    c_totm = np.zeros((steps//15,3))
    c_tot_angm = np.zeros((steps//15,3))
    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
    c_totm[0,:] = total_momentum(v_iter)
    a_iter,rij_betrag = a_vec(r_iter)
    c_energy[0]= energy(r_iter,v_iter,rij_betrag)
    for zähler in range(0,steps):
       
        t_sum= t_sum+delta_t
        if zähler%15==0:
            course_t[zähler//15] =t_sum
        r_iter, v_iter, a_iter,dta_iter, rij_betrag = iterator(r_iter,v_iter,a_iter,delta_t)
        delta_t = time_delta(a_iter,dta_iter)
        if kl(delta_t,t_delta-0.3*t_delta)==True:
            delta_t=t_delta-0.3*t_delta
        elif kl(t_delta+10*t_delta,delta_t)==True:
            delta_t=t_delta+10*t_delta
        if zähler%15==0:
            
            c_energy[zähler//15]= energy(r_iter,v_iter,rij_betrag)
            c_tot_angm[zähler//15,:] = total_angular_momentum(r_iter,v_iter)
            c_totm[zähler//15,:] = total_momentum(v_iter)
            course_r[zähler//15,:,:] =r_iter
            course_v[zähler//15,:,:] =v_iter
        if zähler%50000==0:
            print(zähler)
else:
    t_sum = 0
    delta_t=t_delta
    t_max = 62
    r_iter= r_transformed
    v_iter= v_transformed
    steps = 300000
    course_r = np.zeros((steps//15,N,3))
    course_v = np.zeros((steps//15,N,3))
    course_t = np.zeros((steps//15))
    course_r[0,:,:]=r_iter
    course_v[0,:,:]=v_iter
    course_t[0]= 0
    c_energy = np.zeros((steps//15))
    c_totm = np.zeros((steps//15,3))
    c_tot_angm = np.zeros((steps//15,3))
    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
    c_totm[0,:] = total_momentum(v_iter)
    a,rij_betrag = a_vec(r_iter)
    c_energy[0]= energy(r_iter,v_iter,rij_betrag)
    
    for zähler in range(0,steps-14):
       
        t_sum= t_sum+delta_t
        if zähler%15==0:
            course_t[zähler//15] =t_sum
        r_iter, v_iter, a_iter, dta_iter,rij_betrag = iterator(r_iter,v_iter,delta_t)
        delta_t = time_delta(a_iter,dta_iter)
        if kl(delta_t,t_delta-0.3*t_delta)==True:
            delta_t=t_delta-0.3*t_delta
        elif kl(t_delta+10*t_delta,delta_t)==True:
            delta_t=t_delta+10*t_delta
        if zähler%15==0:
            
            c_energy[zähler//15]= energy(r_iter,v_iter,rij_betrag)
            c_tot_angm[zähler//15,:] = total_angular_momentum(r_iter,v_iter)
            c_totm[zähler//15,:] = total_momentum(v_iter)
            course_r[zähler//15,:,:] =r_iter
            course_v[zähler//15,:,:] =v_iter
        if zähler%5000==0:
            print(zähler)

 




def energy_tot(pos,vel):
    v_sq= np.linalg.norm(vel,axis=2)
    T= np.dot(m[:,0],v_sq*v_sq)/2
    V= np.zeros(N)
    for n in range(N):
        norm = np.linalg.norm(pos-pos[:,n,:],axis=2)
        np.place(norm,norm==0,['inf'])
        V[n]= np.dot(m[:,0],m[n,0]*1/norm)/2
    V = sum(V)
    return T-V

root = pathlib.Path.cwd()
paths =  root / 'e_data.txt', \
root / 'r_data.txt', \
root / 'v_data.txt', \
root / 't_data.txt', \
root / 'angmom_data.txt', \
root / 'mom_data.txt'
for path in paths:
    pathlib.Path.touch(path)

with open(paths[0],'w') as e_file:
    
    np.savetxt(e_file,c_energy, fmt='%0.7e')

with open(paths[1],'w') as r_file:
    save_r = np.reshape(course_r,steps//15*N*3)
    np.savetxt(r_file,save_r, fmt='%.7e')

with open(paths[2],'w') as v_file:
    save_v = np.reshape(course_v,steps//15*N*3)
    np.savetxt(v_file,save_v, fmt='%.7e')

with open(paths[3],'w') as t_file:
    np.savetxt(t_file,course_t, fmt='%.7e')

with open(paths[4],'w') as angmom_file:
    save_am = np.reshape(c_tot_angm,steps//15*3)
    np.savetxt(angmom_file,save_am, fmt='%.7e')

with open(paths[5],'w') as mom_file:
    np.savetxt(mom_file,c_totm, fmt='%.7e')




print(t_sum)
abs_momentum = np.linalg.norm(c_totm,axis=1) 
fig1, ax1 = plt.subplots(1,1)
for i in range(N):
    ax1.plot(course_r[:,i,0],course_r[:,i,1])
ax1.set_ylabel('y-Achse')
ax1.set_xlabel('x-Achse')

fig2, ax2 = plt.subplots(1,1)

ax2.plot(course_t,(abs_momentum-abs_momentum[0]))

ax2.set_xlabel('Time')
ax2.set_ylabel('$(|P(t)|-|P_0|)/|P_0|$')

fig3, ax3  = plt.subplots(1,1)

ax3.plot(course_t,(c_energy-c_energy[0])/c_energy[0])
ax3.set_ylabel('$(E-E_0)/E_0$')
ax3.set_xlabel('Time')
fig4 ,ax4 = plt.subplots(1,1)
ax4.plot(course_t,(c_tot_angm[:,2]-c_tot_angm[0,2])/c_tot_angm[0,2])
ax4.set_xlabel('Time')
ax4.set_ylabel('$(L^z(t)-L^z_0)/L^z_0$')

plt.show()