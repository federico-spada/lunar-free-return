import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize 
from scipy.optimize import NonlinearConstraint, LinearConstraint, BFGS

import spiceypy as spice
import rebound
import assist

### constants
mue = 398600.4 # km^3/s^2
mum = 4902.8 # km^3/s^2
days = 86400. # sec

au = 149597870.700 # km    
RS = 66183. # km 
D = 384.4e3 # km
Re = 6378. # km
Rm = 1737. # km

crd = np.pi/180

def propagate(sv0_,t0,dt,N_times,ref="earth"):
    # extremes of integration
    t_initial = t0 + et0/days
    t_final = t_initial + dt
    # set up rebound simulation with assist extension
    assist_path = '../../rebound_assist/data/'
    ephem = assist.Ephem(assist_path+'linux_p1550p2650.440',
                         assist_path+'sb441-n16.bsp')
    sim = rebound.Simulation()
    # change initial conditions to barycentric frame
    center = ephem.get_particle(ref, t_initial)
    spcrft = center + rebound.Particle(x=sv0_[0]/au, y=sv0_[1]/au, z=sv0_[2]/au,
                      vx=sv0_[3]*days/au, vy=sv0_[4]*days/au, vz=sv0_[5]*days/au )
    sim.add(spcrft)
    # initialize simulation
    sim.t = t_initial
    extras = assist.Extras(sim, ephem)
    sim.ri_ias15.min_dt = 1e-3
    forces = extras.forces
    forces.remove("NON_GRAVITATIONAL")
    extras.forces = forces
    # prepare for integration 
    times = np.linspace(t_initial,t_final,N_times)
    state_vector = np.zeros((N_times,6))
    # run simulation 
    for i, t in enumerate(times):
        extras.integrate_or_interpolate(t)
        earth = ephem.get_particle("earth", t)
        # geocentric output in km, km/s 
        state_vector[i,:3] = ( np.array(sim.particles[0].xyz) 
                           -   np.array(earth.xyz) )*au
        state_vector[i,3:] = ( np.array(sim.particles[0].vxyz) 
                           -   np.array(earth.vxyz) )*au/days
    return state_vector

def setup(x,N1=25,N2=25,N3=25,N4=25): 
    # first segment: trans-lunar injection
    s1 = {'t0':t00, 'dt':x[0], 'rp':rp_tli, 'ecc':0., 'inc':x[1], 'LAN':x[2],
          'arp':0., 'M':x[3], 'DV0':x[4], 'mu':mue, 'Npt':N1, 'ref':"earth"}
    # second segment: trans-earth injection (backward)
    s2 = {'t0':x[5], 'dt':x[6], 'rp':rp_tei, 'ecc':0., 'inc':x[7], 'LAN':x[8],
          'arp':0., 'M':x[9], 'DV0':x[10], 'mu':mue, 'Npt':N2, 'ref':"earth"}
    # third segment: post-perilune, lunar-centered hyperbola
    s3 = {'t0':x[11], 'dt':x[12], 'rp':rp_m, 'ecc':x[13], 'inc':x[14], 'LAN':x[15],
          'arp':x[16], 'M':0., 'DV0':0., 'mu':mum, 'Npt':N3, 'ref':"moon"}    
    # fourth segment: pre-perilune, lunar-centered hyperbola
    s4 = {'t0':x[11], 'dt':x[17], 'rp':rp_m, 'ecc':x[13], 'inc':x[14], 'LAN':x[15],
          'arp':x[16], 'M':0., 'DV0':0., 'mu':mum, 'Npt':N4, 'ref':"moon"} 
    return [s1, s2, s3, s4]
    

def integrate_trajectory(segments):
    y = [] 
    for i, s in enumerate(segments):
        elts = (s['rp'], s['ecc'], s['inc'], s['LAN'], s['arp'], s['M'], 
                s['t0'], s['mu'])
        y0 = spice.conics(elts, s['t0'])
        y0[3:6] = y0[3:6]*( 1. + s['DV0']/np.linalg.norm(y0[3:6]) )
        ys = propagate(y0,s['t0'],s['dt'],s['Npt'],ref=s['ref'])
        y.append(ys)   
    return y  

def obj_fun(x):
    return x[5]

def obj_jac(x):
    oj = np.zeros(len(x))
    oj[5] = 1.0
    return oj

def obj_hess(x):
    return np.zeros(len(x))

def c_eq(x):
    segments = setup(x)
    sv1, sv2, sv3, sv4 = integrate_trajectory(segments)
    return np.r_[ sv1[-1,:]-sv4[-1,:], sv2[-1,:]-sv3[-1,:] ]

def callback(x,ss):
    if np.mod(ss.nit, 10) == 0:
       display(x)

def display(x):
    xd = x.copy()
    xd[[1,2,3,7,8,9,14,15,16]] = np.mod(x[[1,2,3,7,8,9,14,15,16]]/crd, 360)
    print('===================================================================')  
    print('              s1              s2              s3              s4')
    print('===================================================================')
    print(('t0 '+4*'%16.5f') % (t00,xd[5], xd[11], xd[11]))
    print(('dt '+4*'%16.5f') % (xd[0], xd[6], xd[12], xd[17]))
    print(('q  '+4*'%16.1f') % (rp_tli, rp_tei, rp_m , rp_m ))
    print(('e  '+4*'%16.5f') % (0.0, 0.0, xd[13], xd[13]))
    print(('i  '+4*'%16.5f') % (xd[1], xd[7], xd[14], xd[14]))
    print(('Ω  '+4*'%16.5f') % (xd[2], xd[8], xd[15], xd[15]))
    print(('ω  '+4*'%16.5f') % (0.0, 0.0, xd[16], x[16]))
    print(('M  '+4*'%16.5f') % (xd[3], xd[9], 0.0 , 0.0 ))
    print(('∆V0%16.5f%16.5f         N/A             N/A') % (xd[4], xd[10]))
    print('===================================================================')






if __name__ == '__main__':

    ### load SPICE Kernels through meta-Kernel file
    spice.furnsh('spice.mkn')

    ### reference epoch 
    epoch0 = '2009-4-3 00:00:00 UTC'
    et0 = spice.str2et(epoch0)
    #epoch0 = '2020-5-4 12:00:00 UTC'
    #et0 = spice.str2et(epoch0) - 2.888*days

    # fixed parameters 
    t00 = 0.0
    rp_tli  = Re + 200
    rp_tei  = Re + 200 
    rp_m  = Rm + 8263

    ### initial values of search parameters
    x0 = np.array([ 1.0, 0.0, 0.0, 0.0, 3.2,  8.0, -1.0, np.pi/4, 0.0, 0.0, 3.2,
                    4.0, 1.0, 1.0, 0.0, np.pi, 0.0,  -1.0 ])

    #x0 = np.array([ 1.88838209,  0.66678157, -0.2259904 , -0.0530883 ,  3.14203396,
    #    7.59221133, -1.88240663,  0.11505741,  0.98316457, -1.42867853,
    #    3.13799639,  3.61242107,  2.09738364,  2.66275911,  2.65398911,
    #    2.61048846, -0.15247821, -1.72403897])

    ### constraints 
    # non-linear, equality (continuity at s1--s4, s2--s3 boundaries)  
    nln_eq_con = NonlinearConstraint(c_eq, 0., 0., jac='2-point', hess=BFGS())
    # linear, equality (TOF at at s1--s4, s2--s3 boundaries)
    Aeq = np.zeros((2,18))
    Aeq[0,0], Aeq[0,11], Aeq[0,17] = 1., -1., -1.
    Aeq[1,5], Aeq[1,6], Aeq[1,11], Aeq[1,12] = 1., 1., -1., -1.
    lin_eq_cn = LinearConstraint( Aeq, np.repeat(0.,2), np.repeat(0.,2))
    # linear, inequality (minimum TOF's of segments)
    Ane = np.zeros((4,18))
    Ane[0,0], Ane[1,6], Ane[2,12], Ane[3,17] = 1., -1., 1., -1.
    lin_ne_cn = LinearConstraint( Ane, np.repeat(1.,4), np.repeat(np.inf,4) )
    cons = (lin_ne_cn, lin_eq_cn, nln_eq_con)

    ### optimizer options
    opts = {'disp': True, 'verbose': 2, 'maxiter': 5, 
            'factorization_method':'SVDFactorization'}

    ### call the optimizer
    res = minimize( obj_fun, x0, jac=obj_jac, hess=obj_hess, method='trust-constr', 
                    options=opts, constraints=cons, )
    xf = res.x
    #xf = x0

    display(xf)

    ### plot trajectory
    # construct optimized trajectory
    ss = setup(xf,N1=150,N2=150,N3=50,N4=50)
    sv1, sv2, sv3, sv4 = integrate_trajectory(ss)
    etm = et0 + np.linspace(0,xf[5])*days 
    rm, _ = spice.spkpos('301', etm, 'J2000', 'NONE', '399')
    plt.ion()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(0,0,0,'o')
    ax.plot3D(rm[:,0],rm[:,1],rm[:,2],'-',color='darkgray',label='Moon')
    ax.plot3D(sv1[:,0],sv1[:,1],sv1[:,2],label='s1')
    ax.plot3D(sv2[:,0],sv2[:,1],sv2[:,2],label='s2')
    ax.plot3D(sv3[:,0],sv3[:,1],sv3[:,2],label='s3')
    ax.plot3D(sv4[:,0],sv4[:,1],sv4[:,2],label='s4')
    ax.legend()
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    plt.show()
     

    ### unload Kernel 
    spice.kclear()
 
