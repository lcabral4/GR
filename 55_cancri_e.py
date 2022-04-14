#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:13:15 2022

@author: lcabral4
"""

#
# Program 4.3: Precession of Mercury (mercury.ipynb)
# J Wang, Computational modeling and visualization with Python
#
from scipy.stats import linregress
import numpy as np         # get ODE solvers, numpy
import vpython as vp            # get VPython modules for animation
import matplotlib.pyplot as plt # get matplotlib plot functions
import sys
import numpy as np


vec=vp.vector

def leapfrog_tt(lfdiffeq, r0, v0, t0, w0, h):
    """ vectorized leapfrog_tt with time transformation,
        Omega=1/r, that solves general (r,v) ODEs as:
        dr[i]/dt = f[i](v), and dv[i]/dt = g[i](r).
        User supplied lfdiffeq(id, r, v, t) returns
        f[i](r) if id=0, or g[i](v) if id=1 """
    # 1st step: calc r at h/2
    hw = h/(2.0*w0)                         # half h/2w0
    t1 = t0 + hw                      
    r1 = r0 + hw*lfdiffeq(0, r0, v0, t0)    # id=0, get $\vec{r}_{1/2}$
    r2 = np.dot(r1, r1)                     # get r^2=x*x+y*y+z*z
    r12 = np.sqrt(r2)                       # $r_{1/2}$
   
    # 2nd step: calc v1 using r at h/2
    v1 = v0 + h*r12*lfdiffeq(1, r1, v0, t1) # id=1 for g(r) at h/2
    rdotv = np.dot(r1, v0+v1)/2.            # $\vec{r}\cdot\vec{v}_{1/2}$
    w1 = w0 - rdotv*h/r2                    # $ w_0 - \vec{r}\cdot\vec{v}_{1/2} h /r^2$
       
    # 3rd step: calc r by another 1/2 step using v1
    hw = h/(2.0*w1)
    t1 = t1 + hw
    r1 = r1 + hw*lfdiffeq(0, r1, v1, t1)    # get $\vec{r}_{1}$ at t+h
    return r1, v1, t1, w1


#from sympy import *
from numpy import *

G = 6.67*10**(-11)
M = 1.8*10**(30)
c = 3*10**8
e = 0.05
#a = 58000000000
a = 0.01544*1.496*10**11
h = G*3*M*a*(1-e**2)/ c**2
f = h* 4.4684e-23
print(f, 'lambda for 55 Cancri e')

G = 4*pi**2
a = 0.01544
e = 0.05
M = 0.91

rmax = a*(1+e)
Vap = (G*M*(1-e) / (a*(1+e)))**(0.5)
print(rmax, 'rmax 55 Cancri e')
print(Vap, 'Vap 55 Cancri e')

def mercury(id, r, v, t):       # eqns of motion for mercury
    if (id == 0): return v      # velocity, dr/dt
    s = vp.mag(vec(r[0],r[1],r[2]))
    return -GM*r*(1.0 + lamb/(s*s))/(s*s*s)#*ven_merc_mass     # acceleration, dv/dt

def set_scene(r):     # r = init position of planet
    # draw scene, mercury, sun, info box, Runge-Lenz vector
    scene = vp.canvas(title='Precession of 55 Cancri e', 
                       center=vec(.1*0,0,0), background=vec(.2,.5,1))
    planet= vp.sphere(pos=r, color=vec(.9,.6,.4), make_trail=True, radius=0.05)
    sun   = vp.sphere(pos=vec(0,0,0), color=vp.color.yellow, radius=0.02)
    sunlight = vp.local_light(pos=vec(0,0,0), color=vp.color.yellow)
    info = vp.label(pos=vec(.3,-.4,0), text='Angle') # angle info
    RLvec = vp.arrow(pos=vec(0,0,0), axis=vec(-1,0,0), length = 0.25)
    return planet, info, RLvec

time = []
distance = []
def go(animate = False):                     # default: True
    r, v = np.array([0.016212, 0.0, 0.]), np.array([0.0, 45.9 , 0.]) # init r, v for merc
    #r, v = np.array([0.7282, 0.0, 0.]),  np.array([0.0, 7.338, 0.]) # init r, v for ven
    #r, v = np.array([1.1067, 0.0, 0.]),  np.array([0.0, 6.179, 0.]) # init r, v for earth
    t, h, ta, angle = 0.0, 0.002, [], []
    rvec=vec(r[0],r[1],r[2])
    w = 1.0/vp.mag(rvec)                       # $W_0=\Omega(r)$
    
    if (animate): planet, info, RLvec = set_scene(rvec)
    while 0<=t<=100:  # run for 100 years
        rvec=vec(r[0],r[1],r[2])
        vvec=vec(v[0],v[1],v[2])
        L = vp.cross(rvec, vvec)            # $\vec{L}/m=\vec{r}\times \vec{v}$
        A = vp.cross(vvec, L) - GM*rvec/vp.mag(rvec) # scaled RL vec, 
        ta.append(t)
        angle.append(np.arctan(A.y/A.x)*180*3600/np.pi) # arcseconds
        
        if (animate):    
            vp.rate(100)   
            planet.pos = rvec                           # move planet.1
            RLvec.axis, RLvec.length = A, .25           # update RL vec
            info.text='Angle": %8.2f' %(angle[-1])      # angle info 
        r, v, t, w = leapfrog_tt(mercury, r, v, t, w, h)
    
    
    trial = range(0,415)
    m = []
    test = []
    for k in trial:
        for i,j in zip(ta,angle):
            if (0 + k*0.241) <= i <= (0.241 + k*0.241):
                m.append(j)
        f = [m]
            
        test.append(np.array(m[k:]))

    line = []
    for i in test:
        #print(max(i))
        line.append(max(i))
    maximum = max(line)
    minimum = min(line)
    # y = mx + b
    m = (maximum - minimum) / 100
    print(m, 'This is the slope')
    x = np.linspace(0,100,415)
    print(minimum, 'This is the y intercept')
    precession = maximum - minimum
    print(precession, 'This is the precession rate over a century')
    

    plt.figure(1)        # make plot
    plt.title('Precession Rate of 55 Cancri e Due to GR Over a Century')
    plt.plot(ta, angle)
    plt.plot(x,line, label='Precession Line')
    plt.legend()
    plt.xlabel('Time (year)'), plt.ylabel('Precession (arcsec)')
    plt.savefig('55_Cancri_e.pdf')
    plt.show()



GM = 4*np.pi*np.pi      # G*Msun
# lamb=relativistic correction, global, used in 'mercury()'
lamb = 4.120224889981941E-10 #Earth
# lamb = 2.1E-8 #Venus
# lamb = 2.9E-8 #Earth
'''
if (sys.version_info[0] < 3):
    input('Please enter lambda, eg: 0.01, or 1.1E-8 :> ')
else:
    eval(input('Please enter lambda, eg: 0.01, or 1.1E-8 :> '))'''
go(animate = False)