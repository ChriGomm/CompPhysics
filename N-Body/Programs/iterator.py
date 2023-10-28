import numpy as np
import scipy
from numba import njit, vectorize, int32, int64, float32, float64
import csv





for size in range(2):
    if size ==1:
        with open("pl_100.txt", "r") as datafile:
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

        m = m/sum(m)
        R = sum(m*r)
        RV = sum(m*v)
        r_transformed = r-R
        v_transformed = v-RV
        delta_t = t_delta

        @njit
        def acc_vec(pos):
            a = np.zeros((N,3))
            for i in range(N):
                store = np.zeros((N,3))
                for j in range(N):
                    if i!=j:
                        store[j,:] = (m[j,0]*(pos[j,:]-pos[i,:])/np.sum((pos[j,:]-pos[i,:])**2,axis=0)**(3/2))
                a[i,:] =np.sum(store,axis=0)
            return a, a

            
        @njit
        def dot(a,b):
            for i in range(3):
                return (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])
        # @njit
        # def acc_der(pos,vel):
        #     a = np.zeros((N,3))
        #     for i in range(0,N):
        #         store = np.zeros((N,3))
        #         for j in range(0,N):
        #             if i!=j:
        #                 store[j,:] = m[j,0]*((vel[j,:]-vel[i,:])/dot(pos[j,:]-pos[i,:],pos[j,:]-pos[i,:])**(3/2)-3*dot(vel[j,:]-vel[i,:],pos[j,:]-pos[i,:])*(pos[j,:]-pos[i,:])/dot(pos[j,:]-pos[i,:],pos[j,:]-pos[i,:])**(5/2))
        #         a[i,:] = np.sum(store,axis=0)
        #     return a
        @njit 
        def acc_der(pos,vel):
            a = np.zeros((N,3))
            for i in range(0,N):
                store = np.zeros((N,3))
                for j in range(0,N):
                    if i!=j:
                        store[j,:] = (m[j,0]*((vel[j,:]-vel[i,:])/(np.sum((pos[j,:]-pos[i,:])**2)**(3/2))-3*np.sum((vel[j,:]-vel[i,:])*(pos[j,:]-pos[i,:]))*(pos[j,:]-pos[i,:])/(np.sum((pos[j,:]-pos[i,:])**2)**(5/2))))
                a[i,:] = np.sum(store,axis=0)
            return a 
        @njit
        def eng(pos,vel,k):
            v_sq= np.sum(vel*vel,axis=1)
            T= np.sum(m[:,0]*v_sq)/2
            V= np.zeros(N)
            for n in range(N):
                
                norm = np.sqrt(np.sum((pos-pos[n])**2,axis=1))
                norm[n]=1
                V[n]= np.sum(m[:,0]*m[n,0]*1/norm,axis=0)/2-m[n,0]*m[n,0]/2

            V = np.sum(V)
            return T+V
        @njit
        def total_angular_momentum(pos,vel):
            cross = np.zeros((N,3))
            for n in range(N):
                cross[n,0] = pos[n,1]*vel[n,2]-pos[n,2]*vel[n,1]
                cross[n,1] = pos[n,2]*vel[n,0]-pos[n,0]*vel[n,2]
                cross[n,2] = pos[n,0]*vel[n,1]-pos[n,1]*vel[n,0]
                cross[n,:] *= m[n,0]
            return np.sum(cross,axis=0)

        @njit
        def total_momentum(vel):
            mom = np.zeros((N,3))
            for n in range(N):
                mom[n,:] = m[n,0]*vel[n,:]
            return np.sum(mom,axis=0)

        @njit 
        def cross(a,b):
            c = np.zeros(3)
            c[0] = a[1]*b[2]-a[2]*b[1]
            c[1] = a[2]*b[0]-a[0]*b[2]
            c[2] = a[0]*b[1]-a[1]*b[0]
            return c

        @njit
        def j_vec(pos,vel):
            r = pos[1,:]-pos[0,:]
            v = vel[1,:]-vel[0,:]
            return cross(r,v), r, v

        @njit
        def runge_lenz(j,r,v):
            return cross(v,j)-r/np.sqrt(np.sum(r**2))

        @njit
        def a_gha(j,e):
            return np.sum(j**2)/(1-np.sum(e**2))

        @njit
        def fac(i):
            result = 1
            for x in range(2,i+1):
                result = result*x
            return result
        @njit
        def test(func1,func2,v,numb):
            return func1(v)/func2(numb)

        zero = np.zeros((N,3))

        @njit
        def euler(pos,vel,tdelt):
            a,rij_betrag = acc_vec(pos)
            dta = acc_der(pos,vel)
            vel1 = vel+a*tdelt
            pos = pos+vel*tdelt
            return pos, vel1, a, dta, rij_betrag
        @njit  
        def euler_cromer(pos,vel,tdelt):
            a,rij_betrag = acc_vec(pos)
            dta = acc_der(pos,vel)
            vel= vel+a*tdelt
            pos = pos+ vel*tdelt
            return pos, vel, a, dta, rij_betrag
        @njit

        def velo_verlet(pos,vel,a1,tdelt):
            
            pos = pos + vel*tdelt+a1*tdelt**2/2
            a2,rij_betrag = acc_vec(pos)
            vel = vel +(a1+a2)*tdelt/2
            return pos, vel, a2, rij_betrag
        
        @njit
        def kick_drift(pos,vel,a1,tdelt):
            dta = acc_der(pos,vel)
            vel1 = vel+a1*tdelt/2
            pos = pos +vel1*tdelt
            a2,rij_betrag = acc_vec(pos)
            vel = vel1 + a2*tdelt/2
            return pos, vel, a2, dta, rij_betrag
        
        # # Interpolationsfunktion benutzt in Hermite-integratoren
        @njit 
        def interp(Ord,tdelt):
            store = np.zeros((3,N,3))
            # for i in range(0,3):
            #     store[i,:,:]= Ord[i]*tdelt**i/fac(i)
            return sum(store)


        
        
        # print(interp(r_transformed,v_transformed,acc_vec(r_transformed),acc_der(r_transformed,v_transformed),tdelt=t_delta)==r_transformed+v_transformed*t_delta+acc_vec(r_transformed)*t_delta**2/2+acc_der(r_transformed,v_transformed)*t_delta**3/6)
        @njit
        def hermite(pos,vel,tdelt):
            a,rij_betrag = acc_vec(pos)
            dta = acc_der(pos,vel)
            # vp = interp(vel,a,dta,tdelt=tdelt)
            vp= vel+a*tdelt+dta*tdelt*tdelt
            # rp = interp(pos,vel,a,dta,tdelt=tdelt)
            rp = pos+vel*tdelt+a*tdelt**2+dta*tdelt**3
            ap,waste = acc_vec(rp)
            dtap = acc_der(rp,vp)
            dt2a = -6*(a-ap)/tdelt**2-2*(2*dta+dtap)/tdelt
            dt3a = 12*(a-ap)/tdelt**3+6*(dta+dtap)/tdelt**2
            # rc = interp(rp,zero,zero,zero,dt2a,dt3a,tdelt=tdelt)
            rc = rp+1/24*dt2a*tdelt**4+1/120*dt3a*tdelt**5
            # vc = interp(vp,zero,zero,dt2a,dt3a,tdelt=tdelt)
            vc = vp+1/6*dt2a*tdelt**3+1/24*dt3a*tdelt**4
            return rc, vc, a, dta, dt2a, dt3a, rij_betrag

        @njit
        def hermite_integrator(pos,vel,a,ap,dta,dtap):
            vc = vel + (a+ap)*delta_t/2+(dtap-dta)*delta_t**2/12
            rc = pos + (vc+vel)*delta_t/2+(ap-a)*delta_t**2/12
            ac,rij_betrag = acc_vec(rc)
            dtac = acc_der(rc,vc)
            return rc, vc, ac, dtac, rij_betrag
        
        @njit
        def iterated_hermite(pos,vel,a, dta,tdelt):
            # vp = interp(vel,a,dta,tdelt=tdelt)
            vp= vel+a*tdelt+dta*tdelt*tdelt
            # rp = interp(pos,vel,a,dta,tdelt=tdelt)
            rp = pos+vel*tdelt+a*tdelt**2+dta*tdelt**3
            ap, waste = acc_vec(rp)
            dtap = acc_der(rp,vp)
            rc = pos
            vc = vel
            dt2a = -6*(a-ap)/tdelt**2-2*(2*dta+dtap)/tdelt
            dt3a = 12*(a-ap)/tdelt**3+6*(dta+dtap)/tdelt**2
            for i  in range(0,2):
                rc, vc, ap, dtap, rij_betrag = hermite_integrator(rc,vc,a,ap,dta,dtap)
            return rc, vc, ap, dtap, dt2a, dt3a, rij_betrag

        @njit 
        def heun(pos,vel,tdelt):
            a, rij_betrag = acc_vec(pos)
            dta = acc_der(pos,vel)
            v1 = a*tdelt
            r1 = vel*tdelt
            v2 = acc_vec(pos+r1)[0]*tdelt
            r2 = (vel+v2)*tdelt
            v_iter=vel+(v1+v2)/2
            r_iter = pos +(r1+r2)/2
            return r_iter,v_iter,a, dta, rij_betrag
        
        @njit
        def rk4(pos,vel,tdelt):
            a, rij_betrag = acc_vec(pos)
            dta = acc_der(pos,vel)
            v1 = a*tdelt
            r1= vel*tdelt
            v2  = acc_vec(pos+r1/2)[0]*tdelt
            r2 = (vel+v1/2)*tdelt
            v3 = acc_vec(pos+r2/2)[0]*tdelt
            r3 = (vel+v2/2)*tdelt
            v4 = acc_vec(pos+r3)[0]*tdelt
            r4 = (vel+v3)*tdelt
            r_iter = pos +(r1+r4)/6+(r2+r3)/3
            v_iter = vel + (v1+v4)/6+(v2+v3)/3
            return r_iter, v_iter, a, dta, rij_betrag
        
        
        # absolute values
        @njit
        def absol(x):
            return np.sqrt(np.sum(x*x,axis=1))

        @njit 
        def time_delta(a_iter,dta_iter):
            dta_abs = absol(dta_iter)
            for i in range(N):
                if dta_abs[i]==0:
                    dta_abs[i]=0.0001    
            return min(absol(a_iter)/dta_abs)
        
        @njit 
        def time_delta_hermite(a,dta,dt2a,dt3a):
            return min((absol(a)*absol(dt2a)+absol(dta)**2)/(absol(dta)*absol(dt3a)+absol(dt2a)**2))**(1/2)*t_delta

        # @njit
        def iterator(integrator_abrv,steps,block_size):
            if (N==2):
                if (integrator_abrv=='hermite'):
                    integrator = hermite
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    dta = acc_der(r_iter,v_iter)
                    print(int(t_max/t_delta*5))
                    for zähler in range(0,steps-14):
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter,v_iter,a,dta,dt2a,dt3a,rij_betrag = integrator(r_iter,v_iter,delta_t)
                        delta_t = time_delta_hermite(a,dta,dt2a,dt3a)
                        if (delta_t<abs(t_delta-0.3*t_delta)):
                            delta_t=abs(t_delta-0.3*t_delta)
                        elif (t_delta+t_delta<delta_t):
                            delta_t=t_delta+t_delta
                        if zähler%block_size==0:
                            
                            c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        
                        if zähler%100000 == 0:
                            print(zähler)

                elif integrator_abrv=='iterated_hermite':
                    integrator = iterated_hermite
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    dta = acc_der(r_iter,v_iter)
                    print(int(t_max/t_delta*5))
                    for zähler in range(0,steps-14):
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter,v_iter,a,dta,dt2a,dt3a, rij_betrag = integrator(r_iter,v_iter,a,dta,delta_t)
                        delta_t = time_delta_hermite(a,dta,dt2a,dt3a)
                        if (delta_t<abs(t_delta-0.3*t_delta)):
                            delta_t=abs(t_delta-0.3*t_delta)
                        elif (t_delta+t_delta<delta_t):
                            delta_t=t_delta+t_delta
                        if zähler%block_size==0:
                            
                            c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        
                        if zähler%100000 == 0:
                            print(zähler)
                elif integrator_abrv == 'velo_verlet':
                    integrator = velo_verlet
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 3000000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a_iter,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    for zähler in range(0,steps):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter,rij_betrag = integrator(r_iter,v_iter,a_iter,delta_t)
                    
                        if zähler%block_size==0:
                            
                            c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%50000==0:
                            print(zähler)


                elif integrator_abrv == 'kick_drift':
                    integrator = kick_drift
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 300000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a_iter,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    for zähler in range(0,steps):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter,dta_iter, rij_betrag = integrator(r_iter,v_iter,a_iter,delta_t)
                        delta_t = time_delta(a_iter,dta_iter)
                        if (delta_t<t_delta-0.3*t_delta):
                            delta_t=t_delta-0.3*t_delta
                        elif (t_delta+10*t_delta<delta_t):
                            delta_t=t_delta+10*t_delta
                        if zähler%block_size==0:
                            
                            c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%50000==0:
                            print(zähler)
                else:
                    if integrator_abrv == 'euler':
                        integrator = euler
                    elif integrator_abrv == 'euler_cromer':
                        integrator = euler_cromer
                    elif integrator_abrv == 'heun':
                        integrator = heun
                    elif integrator_abrv == 'rk4':
                        integrator = rk4
                    
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 300000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    
                    for zähler in range(0,steps-14):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter, dta_iter,rij_betrag = integrator(r_iter,v_iter,delta_t)
                        delta_t = time_delta(a_iter,dta_iter)
                        if delta_t<t_delta-0.3*t_delta:
                            delta_t=t_delta-0.3*t_delta
                        elif t_delta+10*t_delta<delta_t:
                            delta_t=t_delta+10*t_delta
                        if zähler%block_size==0:
                            
                            c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%5000==0:
                            print(zähler)
            else:
                if (integrator_abrv=='hermite'):
                    integrator = hermite
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    dta = acc_der(r_iter,v_iter)
                    print(int(t_max/t_delta*5))
                    for zähler in range(0,steps-14):
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter,v_iter,a,dta,dt2a,dt3a, rij_betrag = integrator(r_iter,v_iter,delta_t)
                        delta_t = time_delta_hermite(a,dta,dt2a,dt3a)
                        if (delta_t<abs(t_delta-0.3*t_delta)):
                            delta_t=abs(t_delta-0.3*t_delta)
                        elif (t_delta+t_delta<delta_t):
                            delta_t=t_delta+t_delta
                        if zähler%block_size==0:
                            
                            # c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            # c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            # c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        
                        if zähler%100000 == 0:
                            print(zähler)

                elif integrator_abrv=='iterated_hermite':
                    integrator = iterated_hermite
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    dta = acc_der(r_iter,v_iter)
                    print(int(t_max/t_delta*5))
                    for zähler in range(0,steps-14):
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter,v_iter,a,dta,dt2a,dt3a, rij_betrag = integrator(r_iter,v_iter,a,dta,delta_t)
                        delta_t = time_delta_hermite(a,dta,dt2a,dt3a)
                        if (delta_t<abs(t_delta-0.3*t_delta)):
                            delta_t=abs(t_delta-0.3*t_delta)
                        elif (t_delta+t_delta<delta_t):
                            delta_t=t_delta+t_delta
                        if zähler%block_size==0:
                            
                            # c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            # c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            # c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        
                        if zähler%100000 == 0:
                            print(zähler)
                elif integrator_abrv == 'velo_verlet':
                    integrator = velo_verlet
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 3000000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a_iter,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    for zähler in range(0,steps):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter,rij_betrag = integrator(r_iter,v_iter,a_iter,delta_t)
                    
                        if zähler%block_size==0:
                            
                            # c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            # c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            # c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%50000==0:
                            print(zähler)


                elif integrator_abrv == 'kick_drift':
                    integrator = kick_drift
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 300000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a_iter,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    for zähler in range(0,steps):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter,dta_iter, rij_betrag = integrator(r_iter,v_iter,a_iter,delta_t)
                        delta_t = time_delta(a_iter,dta_iter)
                        if (delta_t<t_delta-0.3*t_delta):
                            delta_t=t_delta-0.3*t_delta
                        elif (t_delta+10*t_delta<delta_t):
                            delta_t=t_delta+10*t_delta
                        if zähler%block_size==0:
                            
                            # c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            # c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            # c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%50000==0:
                            print(zähler)
                else:
                    if integrator_abrv == 'euler':
                        integrator = euler
                    elif integrator_abrv == 'euler_cromer':
                        integrator = euler_cromer
                    elif integrator_abrv == 'heun':
                        integrator = heun
                    elif integrator_abrv == 'rk4':
                        integrator = rk4
                    else:
                        print("integrator not recognized.")
                        exit()
                    
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 300000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    
                    for zähler in range(0,steps-14):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter, dta_iter,rij_betrag = integrator(r_iter,v_iter,delta_t)
                        delta_t = time_delta(a_iter,dta_iter)
                        if delta_t<t_delta-0.3*t_delta:
                            delta_t=t_delta-0.3*t_delta
                        elif t_delta+10*t_delta<delta_t:
                            delta_t=t_delta+10*t_delta
                        if zähler%block_size==0:
                            
                            # c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            # c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            # c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%5000==0:
                            print(zähler)
            return t_sum, course_r, course_v, course_t, c_energy, c_totm, c_tot_angm #, c_j, c_rulenz



        steps = int(26000*2.2*15.7)
        block_size = 100
        t_sum, course_r, course_v, course_t, c_energy, c_totm, c_tot_angm = iterator('iterated_hermite',steps,block_size)

        with open("100b_iterated_hermite.txt","w") as file:
            for i in range(steps//block_size):
                file.write("{:.6e}\t".format(course_t[i]))
                for j in range(N):
                    for l in range(3):
                        file.write("{:.6e}\t".format(course_r[i,j,l]))
                for j in range(N):
                    for l in range(3):
                        file.write("{:.6e}\t".format(course_v[i,j,l]))
    else:
        with open("pl_1000.txt", "r") as datafile:
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

        m = m/sum(m)
        R = sum(m*r)
        RV = sum(m*v)
        r_transformed = r-R
        v_transformed = v-RV
        delta_t = t_delta

        @njit
        def acc_vec(pos):
            a = np.zeros((N,3))
            for i in range(N):
                store = np.zeros((N,3))
                for j in range(N):
                    if i!=j:
                        store[j,:] = (m[j,0]*(pos[j,:]-pos[i,:])/np.sum((pos[j,:]-pos[i,:])**2,axis=0)**(3/2))
                a[i,:] =np.sum(store,axis=0)
            return a, a

            
        @njit
        def dot(a,b):
            for i in range(3):
                return (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])
        # @njit
        # def acc_der(pos,vel):
        #     a = np.zeros((N,3))
        #     for i in range(0,N):
        #         store = np.zeros((N,3))
        #         for j in range(0,N):
        #             if i!=j:
        #                 store[j,:] = m[j,0]*((vel[j,:]-vel[i,:])/dot(pos[j,:]-pos[i,:],pos[j,:]-pos[i,:])**(3/2)-3*dot(vel[j,:]-vel[i,:],pos[j,:]-pos[i,:])*(pos[j,:]-pos[i,:])/dot(pos[j,:]-pos[i,:],pos[j,:]-pos[i,:])**(5/2))
        #         a[i,:] = np.sum(store,axis=0)
        #     return a
        @njit 
        def acc_der(pos,vel):
            a = np.zeros((N,3))
            for i in range(0,N):
                store = np.zeros((N,3))
                for j in range(0,N):
                    if i!=j:
                        store[j,:] = (m[j,0]*((vel[j,:]-vel[i,:])/(np.sum((pos[j,:]-pos[i,:])**2)**(3/2))-3*np.sum((vel[j,:]-vel[i,:])*(pos[j,:]-pos[i,:]))*(pos[j,:]-pos[i,:])/(np.sum((pos[j,:]-pos[i,:])**2)**(5/2))))
                a[i,:] = np.sum(store,axis=0)
            return a 
        @njit
        def eng(pos,vel,k):
            v_sq= np.sum(vel*vel,axis=1)
            T= np.sum(m[:,0]*v_sq)/2
            V= np.zeros(N)
            for n in range(N):
                
                norm = np.sqrt(np.sum((pos-pos[n])**2,axis=1))
                norm[n]=1
                V[n]= np.sum(m[:,0]*m[n,0]*1/norm,axis=0)/2-m[n,0]*m[n,0]/2

            V = np.sum(V)
            return T+V
        @njit
        def total_angular_momentum(pos,vel):
            cross = np.zeros((N,3))
            for n in range(N):
                cross[n,0] = pos[n,1]*vel[n,2]-pos[n,2]*vel[n,1]
                cross[n,1] = pos[n,2]*vel[n,0]-pos[n,0]*vel[n,2]
                cross[n,2] = pos[n,0]*vel[n,1]-pos[n,1]*vel[n,0]
                cross[n,:] *= m[n,0]
            return np.sum(cross,axis=0)

        @njit
        def total_momentum(vel):
            mom = np.zeros((N,3))
            for n in range(N):
                mom[n,:] = m[n,0]*vel[n,:]
            return np.sum(mom,axis=0)

        @njit 
        def cross(a,b):
            c = np.zeros(3)
            c[0] = a[1]*b[2]-a[2]*b[1]
            c[1] = a[2]*b[0]-a[0]*b[2]
            c[2] = a[0]*b[1]-a[1]*b[0]
            return c

        @njit
        def j_vec(pos,vel):
            r = pos[1,:]-pos[0,:]
            v = vel[1,:]-vel[0,:]
            return cross(r,v), r, v

        @njit
        def runge_lenz(j,r,v):
            return cross(v,j)-r/np.sqrt(np.sum(r**2))

        @njit
        def a_gha(j,e):
            return np.sum(j**2)/(1-np.sum(e**2))

        @njit
        def fac(i):
            result = 1
            for x in range(2,i+1):
                result = result*x
            return result
        @njit
        def test(func1,func2,v,numb):
            return func1(v)/func2(numb)

        zero = np.zeros((N,3))

        @njit
        def euler(pos,vel,tdelt):
            a,rij_betrag = acc_vec(pos)
            dta = acc_der(pos,vel)
            vel1 = vel+a*tdelt
            pos = pos+vel*tdelt
            return pos, vel1, a, dta, rij_betrag
        @njit  
        def euler_cromer(pos,vel,tdelt):
            a,rij_betrag = acc_vec(pos)
            dta = acc_der(pos,vel)
            vel= vel+a*tdelt
            pos = pos+ vel*tdelt
            return pos, vel, a, dta, rij_betrag
        @njit

        def velo_verlet(pos,vel,a1,tdelt):
            
            pos = pos + vel*tdelt+a1*tdelt**2/2
            a2,rij_betrag = acc_vec(pos)
            vel = vel +(a1+a2)*tdelt/2
            return pos, vel, a2, rij_betrag
        
        @njit
        def kick_drift(pos,vel,a1,tdelt):
            dta = acc_der(pos,vel)
            vel1 = vel+a1*tdelt/2
            pos = pos +vel1*tdelt
            a2,rij_betrag = acc_vec(pos)
            vel = vel1 + a2*tdelt/2
            return pos, vel, a2, dta, rij_betrag
        
        # # Interpolationsfunktion benutzt in Hermite-integratoren
        @njit 
        def interp(Ord,tdelt):
            store = np.zeros((3,N,3))
            # for i in range(0,3):
            #     store[i,:,:]= Ord[i]*tdelt**i/fac(i)
            return sum(store)


        
        
        # print(interp(r_transformed,v_transformed,acc_vec(r_transformed),acc_der(r_transformed,v_transformed),tdelt=t_delta)==r_transformed+v_transformed*t_delta+acc_vec(r_transformed)*t_delta**2/2+acc_der(r_transformed,v_transformed)*t_delta**3/6)
        @njit
        def hermite(pos,vel,tdelt):
            a,rij_betrag = acc_vec(pos)
            dta = acc_der(pos,vel)
            # vp = interp(vel,a,dta,tdelt=tdelt)
            vp= vel+a*tdelt+dta*tdelt*tdelt
            # rp = interp(pos,vel,a,dta,tdelt=tdelt)
            rp = pos+vel*tdelt+a*tdelt**2+dta*tdelt**3
            ap,waste = acc_vec(rp)
            dtap = acc_der(rp,vp)
            dt2a = -6*(a-ap)/tdelt**2-2*(2*dta+dtap)/tdelt
            dt3a = 12*(a-ap)/tdelt**3+6*(dta+dtap)/tdelt**2
            # rc = interp(rp,zero,zero,zero,dt2a,dt3a,tdelt=tdelt)
            rc = rp+1/24*dt2a*tdelt**4+1/120*dt3a*tdelt**5
            # vc = interp(vp,zero,zero,dt2a,dt3a,tdelt=tdelt)
            vc = vp+1/6*dt2a*tdelt**3+1/24*dt3a*tdelt**4
            return rc, vc, a, dta, dt2a, dt3a, rij_betrag

        @njit
        def hermite_integrator(pos,vel,a,ap,dta,dtap):
            vc = vel + (a+ap)*delta_t/2+(dtap-dta)*delta_t**2/12
            rc = pos + (vc+vel)*delta_t/2+(ap-a)*delta_t**2/12
            ac,rij_betrag = acc_vec(rc)
            dtac = acc_der(rc,vc)
            return rc, vc, ac, dtac, rij_betrag
        
        @njit
        def iterated_hermite(pos,vel,a, dta,tdelt):
            # vp = interp(vel,a,dta,tdelt=tdelt)
            vp= vel+a*tdelt+dta*tdelt*tdelt
            # rp = interp(pos,vel,a,dta,tdelt=tdelt)
            rp = pos+vel*tdelt+a*tdelt**2+dta*tdelt**3
            ap, waste = acc_vec(rp)
            dtap = acc_der(rp,vp)
            rc = pos
            vc = vel
            dt2a = -6*(a-ap)/tdelt**2-2*(2*dta+dtap)/tdelt
            dt3a = 12*(a-ap)/tdelt**3+6*(dta+dtap)/tdelt**2
            for i  in range(0,2):
                rc, vc, ap, dtap, rij_betrag = hermite_integrator(rc,vc,a,ap,dta,dtap)
            return rc, vc, ap, dtap, dt2a, dt3a, rij_betrag

        @njit 
        def heun(pos,vel,tdelt):
            a, rij_betrag = acc_vec(pos)
            dta = acc_der(pos,vel)
            v1 = a*tdelt
            r1 = vel*tdelt
            v2 = acc_vec(pos+r1)[0]*tdelt
            r2 = (vel+v2)*tdelt
            v_iter=vel+(v1+v2)/2
            r_iter = pos +(r1+r2)/2
            return r_iter,v_iter,a, dta, rij_betrag
        
        @njit
        def rk4(pos,vel,tdelt):
            a, rij_betrag = acc_vec(pos)
            dta = acc_der(pos,vel)
            v1 = a*tdelt
            r1= vel*tdelt
            v2  = acc_vec(pos+r1/2)[0]*tdelt
            r2 = (vel+v1/2)*tdelt
            v3 = acc_vec(pos+r2/2)[0]*tdelt
            r3 = (vel+v2/2)*tdelt
            v4 = acc_vec(pos+r3)[0]*tdelt
            r4 = (vel+v3)*tdelt
            r_iter = pos +(r1+r4)/6+(r2+r3)/3
            v_iter = vel + (v1+v4)/6+(v2+v3)/3
            return r_iter, v_iter, a, dta, rij_betrag
        
        
        # absolute values
        @njit
        def absol(x):
            return np.sqrt(np.sum(x*x,axis=1))

        @njit 
        def time_delta(a_iter,dta_iter):
            dta_abs = absol(dta_iter)
            for i in range(N):
                if dta_abs[i]==0:
                    dta_abs[i]=0.0001    
            return min(absol(a_iter)/dta_abs)
        
        @njit 
        def time_delta_hermite(a,dta,dt2a,dt3a):
            return min((absol(a)*absol(dt2a)+absol(dta)**2)/(absol(dta)*absol(dt3a)+absol(dt2a)**2))**(1/2)*t_delta

        # @njit
        def iterator(integrator_abrv,steps,block_size):
            if (N==2):
                if (integrator_abrv=='hermite'):
                    integrator = hermite
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    dta = acc_der(r_iter,v_iter)
                    print(int(t_max/t_delta*5))
                    for zähler in range(0,steps-14):
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter,v_iter,a,dta,dt2a,dt3a,rij_betrag = integrator(r_iter,v_iter,delta_t)
                        delta_t = time_delta_hermite(a,dta,dt2a,dt3a)
                        if (delta_t<abs(t_delta-0.3*t_delta)):
                            delta_t=abs(t_delta-0.3*t_delta)
                        elif (t_delta+t_delta<delta_t):
                            delta_t=t_delta+t_delta
                        if zähler%block_size==0:
                            
                            c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        
                        if zähler%100000 == 0:
                            print(zähler)

                elif integrator_abrv=='iterated_hermite':
                    integrator = iterated_hermite
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    dta = acc_der(r_iter,v_iter)
                    print(int(t_max/t_delta*5))
                    for zähler in range(0,steps-14):
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter,v_iter,a,dta,dt2a,dt3a, rij_betrag = integrator(r_iter,v_iter,a,dta,delta_t)
                        delta_t = time_delta_hermite(a,dta,dt2a,dt3a)
                        if (delta_t<abs(t_delta-0.3*t_delta)):
                            delta_t=abs(t_delta-0.3*t_delta)
                        elif (t_delta+t_delta<delta_t):
                            delta_t=t_delta+t_delta
                        if zähler%block_size==0:
                            
                            c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        
                        if zähler%100000 == 0:
                            print(zähler)
                elif integrator_abrv == 'velo_verlet':
                    integrator = velo_verlet
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 3000000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a_iter,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    for zähler in range(0,steps):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter,rij_betrag = integrator(r_iter,v_iter,a_iter,delta_t)
                    
                        if zähler%block_size==0:
                            
                            c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%50000==0:
                            print(zähler)


                elif integrator_abrv == 'kick_drift':
                    integrator = kick_drift
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 300000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a_iter,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    for zähler in range(0,steps):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter,dta_iter, rij_betrag = integrator(r_iter,v_iter,a_iter,delta_t)
                        delta_t = time_delta(a_iter,dta_iter)
                        if (delta_t<t_delta-0.3*t_delta):
                            delta_t=t_delta-0.3*t_delta
                        elif (t_delta+10*t_delta<delta_t):
                            delta_t=t_delta+10*t_delta
                        if zähler%block_size==0:
                            
                            c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%50000==0:
                            print(zähler)
                else:
                    if integrator_abrv == 'euler':
                        integrator = euler
                    elif integrator_abrv == 'euler_cromer':
                        integrator = euler_cromer
                    elif integrator_abrv == 'heun':
                        integrator = heun
                    elif integrator_abrv == 'rk4':
                        integrator = rk4
                    
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 300000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    
                    for zähler in range(0,steps-14):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter, dta_iter,rij_betrag = integrator(r_iter,v_iter,delta_t)
                        delta_t = time_delta(a_iter,dta_iter)
                        if delta_t<t_delta-0.3*t_delta:
                            delta_t=t_delta-0.3*t_delta
                        elif t_delta+10*t_delta<delta_t:
                            delta_t=t_delta+10*t_delta
                        if zähler%block_size==0:
                            
                            c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%5000==0:
                            print(zähler)
            else:
                if (integrator_abrv=='hermite'):
                    integrator = hermite
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    dta = acc_der(r_iter,v_iter)
                    print(int(t_max/t_delta*5))
                    for zähler in range(0,steps-14):
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter,v_iter,a,dta,dt2a,dt3a, rij_betrag = integrator(r_iter,v_iter,delta_t)
                        delta_t = time_delta_hermite(a,dta,dt2a,dt3a)
                        if (delta_t<abs(t_delta-0.3*t_delta)):
                            delta_t=abs(t_delta-0.3*t_delta)
                        elif (t_delta+t_delta<delta_t):
                            delta_t=t_delta+t_delta
                        if zähler%block_size==0:
                            
                            # c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            # c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            # c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        
                        if zähler%100000 == 0:
                            print(zähler)

                elif integrator_abrv=='iterated_hermite':
                    integrator = iterated_hermite
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    dta = acc_der(r_iter,v_iter)
                    print(int(t_max/t_delta*5))
                    for zähler in range(0,steps-14):
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter,v_iter,a,dta,dt2a,dt3a, rij_betrag = integrator(r_iter,v_iter,a,dta,delta_t)
                        delta_t = time_delta_hermite(a,dta,dt2a,dt3a)
                        if (delta_t<abs(t_delta-0.3*t_delta)):
                            delta_t=abs(t_delta-0.3*t_delta)
                        elif (t_delta+t_delta<delta_t):
                            delta_t=t_delta+t_delta
                        if zähler%block_size==0:
                            
                            # c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            # c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            # c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        
                        if zähler%100000 == 0:
                            print(zähler)
                elif integrator_abrv == 'velo_verlet':
                    integrator = velo_verlet
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 3000000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a_iter,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    for zähler in range(0,steps):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter,rij_betrag = integrator(r_iter,v_iter,a_iter,delta_t)
                    
                        if zähler%block_size==0:
                            
                            # c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            # c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            # c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%50000==0:
                            print(zähler)


                elif integrator_abrv == 'kick_drift':
                    integrator = kick_drift
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 300000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a_iter,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    for zähler in range(0,steps):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter,dta_iter, rij_betrag = integrator(r_iter,v_iter,a_iter,delta_t)
                        delta_t = time_delta(a_iter,dta_iter)
                        if (delta_t<t_delta-0.3*t_delta):
                            delta_t=t_delta-0.3*t_delta
                        elif (t_delta+10*t_delta<delta_t):
                            delta_t=t_delta+10*t_delta
                        if zähler%block_size==0:
                            
                            # c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            # c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            # c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%50000==0:
                            print(zähler)
                else:
                    if integrator_abrv == 'euler':
                        integrator = euler
                    elif integrator_abrv == 'euler_cromer':
                        integrator = euler_cromer
                    elif integrator_abrv == 'heun':
                        integrator = heun
                    elif integrator_abrv == 'rk4':
                        integrator = rk4
                    else:
                        print("integrator not recognized.")
                        exit()
                    
                    t_sum = 0
                    delta_t=t_delta
                    t_max = 62
                    r_iter= r_transformed
                    v_iter= v_transformed
                    # steps = 300000
                    course_r = np.zeros((steps//block_size,N,3))
                    course_v = np.zeros((steps//block_size,N,3))
                    course_t = np.zeros((steps//block_size))
                    course_r[0,:,:]=r_iter
                    course_v[0,:,:]=v_iter
                    course_t[0]= 0
                    c_energy = np.zeros((steps//block_size))
                    c_totm = np.zeros((steps//block_size,3))
                    c_tot_angm = np.zeros((steps//block_size,3))
                    c_tot_angm[0,:] = total_angular_momentum(r_iter,v_iter)
                    c_totm[0,:] = total_momentum(v_iter)
                    a,rij_betrag = acc_vec(r_iter)
                    c_energy[0]= eng(r_iter,v_iter,rij_betrag)
                    
                    for zähler in range(0,steps-14):
                    
                        t_sum= t_sum+delta_t
                        if zähler%block_size==0:
                            course_t[zähler//block_size] =t_sum
                        r_iter, v_iter, a_iter, dta_iter,rij_betrag = integrator(r_iter,v_iter,delta_t)
                        delta_t = time_delta(a_iter,dta_iter)
                        if delta_t<t_delta-0.3*t_delta:
                            delta_t=t_delta-0.3*t_delta
                        elif t_delta+10*t_delta<delta_t:
                            delta_t=t_delta+10*t_delta
                        if zähler%block_size==0:
                            
                            # c_energy[zähler//block_size]= eng(r_iter,v_iter,rij_betrag)
                            # c_tot_angm[zähler//block_size,:] = total_angular_momentum(r_iter,v_iter)
                            # c_totm[zähler//block_size,:] = total_momentum(v_iter)
                            course_r[zähler//block_size,:,:] =r_iter
                            course_v[zähler//block_size,:,:] =v_iter
                        if zähler%5000==0:
                            print(zähler)
            return t_sum, course_r, course_v, course_t, c_energy, c_totm, c_tot_angm #, c_j, c_rulenz



        block_size = 100
        iterators = ['euler','euler_cromer','heun','rk4','hermite','iterated_hermite']
        iterators_text = ["euler","euler_cromer","heun","rk4","hermite","iterated_hermite"]
        steps = [int(2600*2.2),int(2600*2.2),int(2600*2.2),int(2600*2.2),int(2600*2.2*15.7),int(2600*2.2*15.7)]
        for it in range(6):
            t_sum, course_r, course_v, course_t, c_energy, c_totm, c_tot_angm = iterator(iterators[it],steps[it],block_size)

            with open("1000b_"+iterators_text[it]+".txt","w") as file:
                for i in range(steps[it]//block_size):
                    file.write("{:.6e}\t".format(course_t[i]))
                    for j in range(N):
                        for l in range(3):
                            file.write("{:.6e}\t".format(course_r[i,j,l]))
                    for j in range(N):
                        for l in range(3):
                            file.write("{:.6e}\t".format(course_v[i,j,l]))


