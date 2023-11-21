# Federico Spada --- 2023/11/21

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize 
from scipy.optimize import NonlinearConstraint, LinearConstraint, BFGS
from extensisq import SWAG
import spiceypy as spice

# Main reference:
# Ocampo, C., 2010, "Elements of a Software System for Spacecraft Trajectory 
# Optimization" in Conway, B.A. ed., 2010. Spacecraft trajectory optimization 
# (Vol. 29). Cambridge University Press.
# Note that this version uses mean anomaly instead of true anomaly as one of
# the orbital elements


### constants
mue = 398600.4 # km^3/s^2
mum = 4902.8 # km^3/s^2
days = 86400. # sec

au = 149597870.700 # km    
RS = 66183. # km 
D = 384.4e3 # km
Re = 6378. # km
Rm = 1737. # km


def derivs(t,y):
    # units for this function: km, s, km/s
    et = et0 + t
    r_, v_ = y[:3], y[3:]
    r = np.linalg.norm(r_)
    rm_, _ = spice.spkpos('301', et, 'J2000', 'NONE', '399')
    rm = np.linalg.norm(rm_)
    rms_ = rm_ - r_
    rms = np.linalg.norm(rms_)
    a_ = -mue*r_/r**3 + mum*(rms_/rms**3 - rm_/rm**3) 
    f = np.r_[v_, a_]
    return f


def integrate_trajectory(segments):
    sols = []
    for i, s in enumerate(segments):
        # dictionary s contains the parameters of the segment
        elts = (s['rp'],s['ecc'],s['inc'],s['LAN'],s['arp'],s['M'],s['t0'],s['mu'])
        # construct initial state vector from orbital elements tuple
        y0 = spice.conics(elts, s['t0'])
        # add initial Delta-V from maneuver at t0
        y0[3:6] = y0[3:6]*( 1. + s['DV0']/np.linalg.norm(y0[3:6]) )
        # if elements are relative to Moon, change to geocentric reference
        if s['ref'] == "moon":
            sm, _ = spice.spkezr('301', et0+s['t0']*days, 'J2000', 'NONE', '399')
            y0 = y0 + sm
        # set up interval of integration 
        tspan = [s['t0']*days, (s['t0']+s['dt'])*days]
        # integrate IVP for current segment
        sol = solve_ivp(derivs, tspan, y0, method=method, rtol=rtol, atol=atol)
        # store integrateed segment solution for output 
        sols.append(sol)
    return sols


def setup(x): 
    # first segment: trans-lunar injection
    el1 = {'t0':t00, 'dt':x[0], 'rp':rp_tli, 'ecc':0., 'inc':x[1], 'LAN':x[2],
          'arp':0., 'M':x[3], 'DV0':x[4], 'mu':mue, 'ref':"earth"}
    # second segment: trans-earth injection (backward)
    el2 = {'t0':x[5], 'dt':x[6], 'rp':rp_tei, 'ecc':0., 'inc':x[7], 'LAN':x[8],
          'arp':0., 'M':x[9], 'DV0':x[10], 'mu':mue, 'ref':"earth"}
    # third segment: post-perilune, lunar-centered hyperbola
    el3 = {'t0':x[11], 'dt':x[12], 'rp':rp_m, 'ecc':x[13], 'inc':x[14], 'LAN':x[15],
          'arp':x[16], 'M':0., 'DV0':0., 'mu':mum, 'ref':"moon"}    
    # fourth segment: pre-perilune, lunar-centered hyperbola
    el4 = {'t0':x[11], 'dt':x[17], 'rp':rp_m, 'ecc':x[13], 'inc':x[14], 'LAN':x[15],
          'arp':x[16], 'M':0., 'DV0':0., 'mu':mum, 'ref':"moon"} 
    return [el1, el2, el3, el4]


# function for equality constraints    
def c_eq(x):
    els = setup(x*xm)
    s1, s2, s3, s4 = integrate_trajectory(els)
    return np.r_[ s1.y[:,-1]-s4.y[:,-1], s2.y[:,-1]-s3.y[:,-1] ]

# objective function to be minimized
def obj_fun(x):
    return x[5]*xm[5]

# Jacobian of the objective function 
def obj_jac(x):
    oj = np.zeros(len(x))
    oj[5] = 1.0*xm[5]
    return oj

# Hessian of the objective function
def obj_hess(x):
    return np.zeros(len(x))

# function for screen output
def display(x):
    xd = x.copy() * xm
    # convert angles to degrees
    xd[[2,3,8,9,15,16]] = np.mod(np.rad2deg(xd[[2,3,8,9,15,16]]), 360)
    # inclinations are in [0, 180]
    xd[[1,7,14]] = np.mod(np.rad2deg(xd[[1,7,14]]), 180)
    print('===================================================================')  
    print('              s1              s2              s3              s4')
    print('===================================================================')
    print(('t0 '+4*'%16.5f') % (t00   , xd[5] , xd[11], xd[11]))
    print(('dt '+4*'%16.5f') % (xd[0] , xd[6] , xd[12], xd[17]))
    print(('q  '+4*'%16.1f') % (rp_tli, rp_tei, rp_m  , rp_m  ))
    print(('e  '+4*'%16.5f') % (0.0   , 0.0   , xd[13], xd[13]))
    print(('i  '+4*'%16.5f') % (xd[1] , xd[7] , xd[14], xd[14]))
    print(('Ω  '+4*'%16.5f') % (xd[2] , xd[8] , xd[15], xd[15]))
    print(('ω  '+4*'%16.5f') % (0.0   , 0.0   , xd[16], xd[16]))
    print(('M  '+4*'%16.5f') % (xd[3] , xd[9] , 0.0   , 0.0 ))
    print(('∆V0%16.5f%16.5f         N/A             N/A') % (xd[4], xd[10]))
    print('===================================================================')


# function for plot
def plot_trajectory_3d(x1):
    # construct trajectory from given x
    els = setup(x1*xm)
    s1, s2, s3, s4 = integrate_trajectory(els)
    etm = et0 + np.linspace(0,x1[5]*xm[5])*days
    rm, _ = spice.spkpos('301', etm, 'J2000', 'NONE', '399')
    plt.ion()
    plt.figure()
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(0,0,0,'o')
    ax.plot3D(rm[:,0],rm[:,1],rm[:,2],'-',color='darkgray',label='Moon')
    ax.plot3D(s1.y.T[:,0],s1.y.T[:,1],s1.y.T[:,2],label='s1')
    ax.plot3D(s2.y.T[:,0],s2.y.T[:,1],s2.y.T[:,2],label='s2')
    ax.plot3D(s3.y.T[:,0],s3.y.T[:,1],s3.y.T[:,2],label='s3')
    ax.plot3D(s4.y.T[:,0],s4.y.T[:,1],s4.y.T[:,2],label='s4')
    ax.legend()
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')



if __name__ == '__main__':

    ### load SPICE Kernels through meta-Kernel file
    spice.furnsh('spice.mkn')

    # ODE solver options
    method = SWAG
    rtol = 1e-10
    atol = 1e-13
    
    ### reference epoch 
    epoch0 = '2009-4-3 00:00:00 UTC'
    et0 = spice.str2et(epoch0)

    # fixed parameters 
    t00 = 0.0
    rp_tli  = Re + 200
    rp_tei  = Re + 200 
    rp_m  = Rm + 8263

    ### initial values of search parameters
    x0 = np.array([ 1.0, 0.0, 0.0, 0.0, 3.2,  8.0, -1.0, np.pi/4, 0.0, 0.0, 3.2,
                    4.0, 1.0, 1.0, 0.0, np.pi, 0.0,  -1.0 ])

    # scaling values (modify with caution!)
    xm = np.array([4, np.pi, 2*np.pi, 2*np.pi, 3.3, 12, 4, np.pi, 2*np.pi, 2*np.pi, 3.3,
                   8, 4, 5, np.pi, 2*np.pi, 2*np.pi, 4])

    ### constraints 
    # non-linear, equality (continuity at s1--s4, s2--s3 boundaries)  
    nln_eq_con = NonlinearConstraint(c_eq, 0., 0., jac='2-point', hess=BFGS())
    # linear, equality (TOF at at s1--s4, s2--s3 boundaries)
    Aeq = np.zeros((2,18))
    Aeq[0,0], Aeq[0,11], Aeq[0,17] = 1., -1., -1.
    Aeq[1,5], Aeq[1,6], Aeq[1,11], Aeq[1,12] = 1., 1., -1., -1.
    lin_eq_cn = LinearConstraint( Aeq*xm, np.repeat(0.,2), np.repeat(0.,2))
    # linear, inequality (minimum TOF's of segments)
    Ane = np.zeros((4,18))
    Ane[0,0], Ane[1,6], Ane[2,12], Ane[3,17] = 1., -1., 1., -1.
    lin_ne_cn = LinearConstraint( Ane*xm, np.repeat(1.,4), np.repeat(np.inf,4) )
    cons = (lin_ne_cn, lin_eq_cn, nln_eq_con)

    ### optimizer options
    opts = {'disp': True, 'verbose': 2, 'maxiter': 30}

    ### call the optimizer
    # NOTE: the optimizer will work on normalized quantities, scaled on xm
    res = minimize( obj_fun, x0/xm, jac=obj_jac, hess=obj_hess, method='trust-constr', 
                    options=opts, constraints=cons)
    xf = res.x
    display(xf)
    plot_trajectory_3d(xf)

    ### unload Kernel 
    spice.kclear()
 
