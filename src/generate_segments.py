"""Human Motion Simulator

This script generates the human kinematics data.

The script is based on "A global human walking model with real-time kinematic personification" paper, by R. Boulic, N.M. Thalmann, and D. Thalmann; The Visual Computer, vol.6, pp.344-358, 1990

The model is based on biomechanical experimental data.

Adopted into a Python framework by Mikolaj Czerkawski
from scripts authored by V.C. Chen and Yang Hai

"""

from .radar_helpers import *

import numpy as np
import scipy.interpolate as interp

def get_gait(rv):
    if rv <= 0:
        raise 'Velocity must be positive'
    elif rv < 0.5:
        return 'a'
    elif rv < 1.3:
        return 'b'
    elif rv <= 3:
        return 'c'
    else:
        raise 'Relative velocity must be less than 3'

def generate_segments(forward_motion = True, height = 1.8, rv = 3.0, gait = '', fs = 100, duration = 10.0, radarloc = (0, 10, 0)):
    """Generates human walking kinematics data based on the input parameters.

    Parameters
    ----------
    forward_motion : bool
        Indicates whether the forward motion of the human should be simulated
        
    height: float
        Height of the simulated human target
    
    rv: float
        Relative velocity of the simulated human target
    
    fs: float
        Sampling rate of the simulated traces
    
    duration : float
        Duration of the simulated traces
    
    radarloc : tuple
        Location of the radar receiver (x, y, z)

    Returns
    -------
    numpy array
        a dictionary containing kinematics data of the simulated walking human
    """
    # body segments' length (meter) - relative to human height
    upperleglen = 0.245*height
    lowerleglen = 0.246*height
    Ht = upperleglen + lowerleglen
    
    # spatial characteristics
    rlc = 1.346*np.sqrt(rv) # relative length of a cycle
    
    # temporal characteristics
    # rdc - Real duration of circle
    rdc = rlc/(rv*Ht)
    numcyc = int(np.ceil(duration/rdc))
    T = rdc*numcyc
    
    # total number of pulses   
    numpl = int(np.round(T*fs))
    nt = int(np.round(numpl/numcyc))
    nt += int(nt % 2 == 1)
    
    
    
    return _generate_segments(forward_motion = forward_motion,
                              height = height,
                              rv = rv,
                              nt = nt,
                              numcyc = numcyc,
                              radarloc = radarloc,
                              gait = gait)

def _generate_segments(forward_motion = True, height = 1.8, rv = 3.0, nt = 1024, numcyc = 2, radarloc = (0, 10, 0), gait = ''):
    """Internal legacy version of the generate_segments function, more similar to original MATLAB code

    Parameters
    ----------
    forward_motion : bool
        Indicates whether the forward motion of the human should be simulated
        
    height: float
        Height of the simulated human target
    
    rv: float
        Relative velocity of the simulated human target
    
    nt: int
        Number of pulses per cycle
    
    numcyc : int
        Number of simulated cycles
    
    radarloc : tuple
        Location of the radar receiver (x, y, z)
        
    gait : string
        Indicates a desired gait, if equal to '' then gait is selected to match velocity (as in original)

    Returns
    -------
    numpy array
        a dictionary containing kinematics data of the simulated walking human
    """
    # total number of pulses   
    numpl = nt*numcyc
    
    # body segments' length (meter) - relative to human height
    headlen = 0.130*height
    shoulderlen = (0.259/2)*height
    torsolen = 0.288*height
    hiplen = (0.191/2)*height
    upperleglen = 0.245*height
    lowerleglen = 0.246*height
    footlen = 0.143*height
    upperarmlen = 0.188*height
    lowerarmlen = 0.152*height
    Ht = upperleglen + lowerleglen
    
    # spatial characteristics
    rlc = 1.346*np.sqrt(rv) # relative length of a cycle
    
    # temporal characteristics
    # rdc - Real duration of circle
    # dc - duration of a cycle
    # ds - duration of support
    # rdsmod - relative duration of support
    # T - total time duration 
    rdc = rlc/(rv*Ht)
    dc = rlc/rv
    ds = 0.752*dc-0.143
    dsmod = ds/dc 
    T = rdc*numcyc
    
    # time scaling
    dt = 1.0/nt
    t = np.linspace(0, 1.0-dt, nt)
    t3 = np.linspace(-1.0, 2.0-dt, nt*3)

    # designate gait characteristic - based on Boulic's model
    if gait == '':
        gait = get_gait(rv)

    # Locations of body segments: Appendix A of Boulic's paper
    #      3 translation trajectory coords. give the body segments location
    #      relative to rv

    # calculate vertical translation: offset from the current height (Hs) of 
    # the origin of the spine Os
    av = 0.015*rv
    verttrans = -av+av*np.sin(2*np.pi*(2*t-0.35))
    maxvl = max(verttrans)
    minvl = min(verttrans)
    diffvl = maxvl-minvl

    # calculate lateral translation: Os oscillates laterally to ensure the
    # weight transfer from one leg to the other.
    if gait == 'a':
        al = -0.128*rv**2+0.128*rv
    else:
        al = -0.032
    lattrans = al*np.sin(2*np.pi*(t-0.1))
    maxvl = max(lattrans)
    minvl = min(lattrans)
    diffvl = maxvl-minvl

    # calculate translation forward/backward: acceleration and deceleration
    # phases. When rv grows this effect decreases. 
    if gait == 'a':
        aa = -0.084*rv**2+0.084*rv
    else:
        aa = -0.021
    phia = 0.625-dsmod
    transforback = aa*np.sin(2*np.pi*(2*t+2*phia))
    maxvl = max(transforback)
    minvl = min(transforback)
    diffvl = maxvl-minvl

    # two rotations of the pelvis- appendix B of Boulic's paper

    # calculate rotation forward/backward: to make forward motion of the leg,
    # the center of gravity of the body must move. To do this, flexing movement
    # of the back relatively to the pelvis must be done.
    if gait == 'a':
        a1 = -8*rv**2+8*rv
    else:
        a1 = 2
    rotforback = -a1+a1*np.sin(2*np.pi*(2*t-0.1))
    maxvl = max(rotforback)
    minvl = min(rotforback)
    diffvl = maxvl-minvl

    # calculate rotation left/right: the pelvis falls on th side of the
    # swinging leg.
    a2 = 0.01*rv
    temp1 = -a2+a2*np.cos(2*np.pi*(10*t[:round(nt*0.15)]/3))
    temp2 = -a2-a2*np.cos(2*np.pi*(10*(t[round(nt*0.15):round(nt*0.50)]-0.15)/7))
    temp3 = a2-a2*np.cos(2*np.pi*(10*(t[round(nt*0.50):round(nt*0.65)]-0.5)/3))
    temp4 = a2+a2*np.cos(2*np.pi*(10*(t[round(nt*0.65):nt]-0.65)/7))
    rotleftright = np.concatenate([temp1,temp2,temp3,temp4])
    maxvl = max(rotleftright)
    minvl = min(rotleftright)
    diffvl = maxvl-minvl

    # calculate torsion rotation: pelvis rotates relatively to the snp.pine to
    # perform the step
    a3 = 4*rv
    torrot = -a3*np.cos(2*np.pi*t)
    maxvl = max(torrot)
    minvl = min(torrot)
    diffvl = maxvl-minvl
    
    # leg flexing/extension: at the hip, at the knee, and at the ankle.

    # calculate flexing at the hip - appendix C of Boulic's paper
    if gait == 'a':
        x1 = -0.1
        x2 = 0.5
        x3 = 0.9
        y1 = 50*rv
        y2 = -30*rv
        y3 = 50*rv
    if gait == 'b':
        x1 = -0.1
        x2 = 0.5
        x3 = 0.9
        y1 = 25
        y2 = -15
        y3 = 25
    if gait == 'c':
        x1 = 0.2*(rv-1.3)/1.7-0.1
        x2 = 0.5
        x3 = 0.9
        y1 = 5*(rv-1.3)/1.7+25
        y2 = -15
        y3 = 6*(rv-1.3)/1.7+25

    if x1+1 == x3:
        x = [x1-1,x2-1,x1,x2,x3,x2+1,x3+1]
        y = [y1,y2,y1,y2,y3,y2,y3]
    else:
        x = [x1-1,x2-1,x3-1,x1,x2,x3,x1+1,x2+1,x3+1]
        y = [y1,y2,y3,y1,y2,y3,y1,y2,y3]
        
    int_obj = interp.PchipInterpolator(x, y)
    temp = int_obj(t3) # cubic interpolation of the control points
    flexhip = temp[nt:2*nt]
    maxvl = max(flexhip)
    minvl = min(flexhip)
    diffvl = maxvl-minvl

    # calculate flexing at the knee: there are 4 control points.
    if gait == 'a':# values from the plots on Boulic's paper
        x1 = 0.17
        x2 = 0.4
        x3 = 0.75
        x4 = 1
        y1 = 3
        y2 = 3
        y3 = 140*rv
        y4 = 3
    if gait == 'b': # values from the plots on Boulic's paper
        x1 = 0.17
        x2 = 0.4
        x3 = 0.75
        x4 = 1
        y1 = 3
        y2 = 3
        y3 = 70
        y4 = 3
    if gait == 'c': # values from the plots on Boulic's paper
        x1 = -0.05*(rv-1.3)/1.7+0.17
        x2 = 0.4
        x3 = -0.05*(rv-1.3)/1.7+0.75
        x4 = -0.03*(rv-1.3)/1.7+1
        y1 = 22*(rv-1.3)/1.7+3
        y2 = 3
        y3 = -5*(rv-1.3)/1.7+70
        y4 = 3*(rv-1.3)/1.7+3

    x = [x1-1,x2-1,x3-1,x4-1,x1,x2,x3,x4,x1+1,x2+1,x3+1,x4+1]
    y = [y1,y2,y3,y4,y1,y2,y3,y4,y1,y2,y3,y4]
    
    int_obj = interp.PchipInterpolator(x, y)
    temp = int_obj(t3) # cubic interpolation of the control points
    flexknee = temp[nt:2*nt]
    maxvl = max(flexknee)
    minvl = min(flexknee)
    diffvl = maxvl-minvl

    # calculate flexing at the ankle: there are 5 control points
    if gait == 'a': # values from the plots on Boulic's paper
        x1 = 0
        x2 = 0.08
        x3 = 0.5
        x4 = dsmod
        x5 = 0.85
        y1 = -3
        y2 = -30*rv-3
        y3 = 22*rv-3
        y4 = -34*rv-3
        y5 = -3
    if gait == 'b':# values from the plots on Boulic's paper
        x1 = 0
        x2 = 0.08
        x3 = 0.5
        x4 = dsmod
        x5 = 0.85
        y1 = -3
        y2 = -18
        y3 = 8
        y4 = -20
        y5 = -3
    if gait == 'c':# values from the plots on Boulic's paper
        x1 = 0
        x2 = 0.08
        x3 = -0.1*(rv-1.3)/1.7+0.5
        x4 = dsmod
        x5 = 0.85
        y1 = 5*(rv-1.3)/1.7-3
        y2 = 4*(rv-1.3)/1.7-18
        y3 = -3*(rv-1.3)/1.7+8
        y4 = -8*(rv-1.3)/1.7-20
        y5 = 5*(rv-1.3)/1.7-3
    x = [x1-1,x2-1,x3-1,x4-1,x5-1,x1,x2,x3,x4,x5,x1+1,x2+1,x3+1,x4+1,x5+1]
    y = [y1,y2,y3,y4,y5,y1,y2,y3,y4,y5,y1,y2,y3,y4,y5]
    
    int_obj = interp.PchipInterpolator(x, y)
    temp = int_obj(t3) # cubic interpolation of the control points
    flexankle = temp[nt:2*nt]
    maxvl = max(flexankle)
    minvl = min(flexankle)
    diffvl = maxvl-minvl

    # trajectory of upper body

    # calculate motion (torsion) of the thorax: there are 4 control points # values from the plots on Boulic's paper - Appendix D
    x1 = 0.1
    x2 = 0.4
    x3 = 0.6
    x4 = 0.9
    y1 = (4/3)*rv
    y2 = (-4.5/3)*rv
    y3 = (-4/3)*rv
    y4 = (4.5/3)*rv
    x = [x1-1,x2-1,x3-1,x4-1,x1,x2,x3,x4,x1+1,x2+1,x3+1,x4+1]
    y = [y1,y2,y3,y4,y1,y2,y3,y4,y1,y2,y3,y4]
    
    int_obj = interp.PchipInterpolator(x, y)
    temp = int_obj(t3) #cubic interpolation of the control points
    motthor = temp[nt:2*nt]
    maxvl = max(motthor)
    minvl = min(motthor)
    diffvl = maxvl-minvl

    # calculate flexing at the shoulder
    ash = 9.88*rv
    flexshoulder = 3-ash/2-ash*np.cos(2*np.pi*t)
    maxvl = max(flexshoulder)
    minvl = min(flexshoulder)
    diffvl = maxvl-minvl

    # calculate flexing at the elbow
    if gait == 'a':
        x1 = 0.05
        x2 = 0.5
        x3 = 0.9
        y1 = 6*rv+3
        y2 = 34*rv+3
        y3 = 10*rv+3
    if gait == 'b':
        x1 = 0.05
        x2 = 0.01*(rv-0.5)/0.8+0.5
        x3 = 0.9
        y1 = 8*(rv-0.5)/0.8+6
        y2 = 24*(rv-0.5)/0.8+20
        y3 = 9*(rv-0.5)/0.8+8
    if gait == 'c':
        x1 = 0.05
        x2 = 0.04*(rv-1.3)/1.7+0.51
        x3 = -0.1*(rv-1.3)/1.7+0.9
        y1 = -6*(rv-1.3)/1.7+14
        y2 = 26*(rv-1.3)/1.7+44
        y3 = -6*(rv-.13)/1.7+17
    x = [x1-1,x2-1,x3-1,x1,x2,x3,x1+1,x2+1,x3+1]
    y = [y1,y2,y3,y1,y2,y3,y1,y2,y3]
    
    int_obj = interp.PchipInterpolator(x, y)
    temp = int_obj(t3)
    flexelbow = temp[nt:2*nt]
    maxvl = max(flexelbow)
    minvl = min(flexelbow)
    diffvl = maxvl-minvl

    # initialization of storage
    ltoe = np.zeros([3, nt])
    rtoe = np.zeros([3, nt])
    lankle = np.zeros([3, nt])
    rankle = np.zeros([3, nt])
    lknee = np.zeros([3, nt])
    rknee = np.zeros([3, nt])
    lhip = np.zeros([3, nt])
    rhip = np.zeros([3, nt])
    lhand = np.zeros([3, nt])
    rhand = np.zeros([3, nt])
    lelbow = np.zeros([3, nt])
    relbow = np.zeros([3, nt])
    lshoulder = np.zeros([3, nt])
    rshoulder = np.zeros([3, nt])
    head = np.zeros([3, nt])
    neck = np.zeros([3, nt])
    
    # Handling lower body flexing at ankles, knees, and hips
    # handle flexing at the left ankle # left and right ankle are in opposite phase
    the = np.zeros(nt)
    psi = np.linspace(0,0,nt)
    the[:round(nt/2)] = flexankle[round(nt/2):nt]*np.pi/180
    the[round(nt/2):nt] = flexankle[:round(nt/2)]*np.pi/180
    phi = np.linspace(0,0,nt)
    for i in range(nt):
        Rxyz = XYZConvention(psi[i],the[i],phi[i])
        temp = np.dot(Rxyz,[footlen,0,0])
        ltoe[0,i] = temp[0] #x,y and z coordinates
        ltoe[1,i] = temp[1]+hiplen
        ltoe[2,i] = temp[2]-(upperleglen+lowerleglen)

    # handle flexing at the right ankle
    psi = np.linspace(0,0,nt)
    the = flexankle*np.pi/180
    phi = np.linspace(0,0,nt)
    for i in range(nt):
        Rxyz = XYZConvention(psi[i],the[i],phi[i])
        temp = np.dot(Rxyz,[footlen,0,0])
        rtoe[0,i] = temp[0] #x,y and z coordinates
        rtoe[1,i] = temp[1]-hiplen
        rtoe[2,i] = temp[2]-(upperleglen+lowerleglen)

    # handle flexing at the left knee
    psi = np.linspace(0,0,nt)
    the[:round(nt/2)] = flexknee[round(nt/2):nt]*np.pi/(-180)
    the[round(nt/2):nt] = flexknee[:round(nt/2)]*np.pi/(-180)
    phi = np.linspace(0,0,nt)
    for i in range(nt):
        Rxyz = XYZConvention(psi[i],the[i],phi[i]) 
        temp = np.dot(Rxyz,[0,0,-lowerleglen])
        lankle[0,i] = temp[0] #x,y and z coordinates
        lankle[1,i] = temp[1]+hiplen
        lankle[2,i] = temp[2]-upperleglen
        temp = np.dot(Rxyz,[ltoe[0,i],0,ltoe[2,i]+upperleglen])
        ltoe[0,i] = temp[0] # updates x,y and z coordinates of the left toe
        ltoe[1,i] = temp[1]+hiplen
        ltoe[2,i] = temp[2]-upperleglen

    # handle flexing at the right knee
    psi = np.linspace(0,0,nt)
    the = flexknee*np.pi/(-180)
    phi = np.linspace(0,0,nt)
    for i in range(nt):
        Rxyz = XYZConvention(psi[i],the[i],phi[i])
        temp = np.dot(Rxyz,[0,0,-lowerleglen])
        rankle[0,i] = temp[0]
        rankle[1,i] = temp[1]-hiplen
        rankle[2,i] = temp[2]-upperleglen
        temp = np.dot(Rxyz,[rtoe[0,i],0,rtoe[2,i]+upperleglen])
        rtoe[0,i] = temp[0] # updates x,y and z coordinates of the right toe
        rtoe[1,i] = temp[1]-hiplen
        rtoe[2,i] = temp[2]-upperleglen

    # handle flexing at the left hip
    psi = np.linspace(0,0,nt)
    the[:round(nt/2)] = flexhip[round(nt/2):nt]*np.pi/180
    the[round(nt/2):nt] = flexhip[:round(nt/2)]*np.pi/180
    phi = np.linspace(0,0,nt)
    for i in range(nt):
        Rxyz = XYZConvention(psi[i],the[i],phi[i])
        temp = np.dot(Rxyz,[0,0,-upperleglen])
        lknee[0,i] = temp[0] #  x,y and z coordinates
        lknee[1,i] = temp[1]+hiplen
        lknee[2,i] = temp[2]
        temp = np.dot(Rxyz,[lankle[0,i],0,lankle[2,i]])# updates x,y and z coordinates of the left ankle
        lankle[0,i] = temp[0] 
        lankle[1,i] = temp[1]+hiplen
        lankle[2,i] = temp[2]
        temp = np.dot(Rxyz,[ltoe[0,i],0,ltoe[2,i]])# updates x,y and z coordinates of the left toe
        ltoe[0,i] = temp[0]
        ltoe[1,i] = temp[1]+hiplen
        ltoe[2,i] = temp[2]

    # handle flexing at the right hip
    psi = np.linspace(0,0,nt)
    the = flexhip*np.pi/180
    phi = np.linspace(0,0,nt)
    for i in range(nt):
        Rxyz = XYZConvention(psi[i],the[i],phi[i])
        temp = np.dot(Rxyz,[0,0,-upperleglen])
        rknee[0,i] = temp[0] # x,y and z coordinates
        rknee[1,i] = temp[1]-hiplen
        rknee[2,i] = temp[2]
        temp = np.dot(Rxyz,[rankle[0,i],0,rankle[2,i]])
        rankle[0,i] = temp[0]# updates x,y and z coordinates of the right ankle
        rankle[1,i] = temp[1]-hiplen
        rankle[2,i] = temp[2]
        temp = np.dot(Rxyz,[rtoe[0,i],0,rtoe[2,i]]) # updates x,y and z coordinates of the right toe
        rtoe[0,i] = temp[0]
        rtoe[1,i] = temp[1]-hiplen
        rtoe[2,i] = temp[2]

    # Handling lower body rotation
    psi = rotleftright*np.pi/(-180)
    the = np.linspace(0,0,nt)
    phi = torrot*np.pi/180
    for i in range(nt):
        Rxyz = XYZConvention(psi[i],the[i],phi[i])
        temp = np.dot(Rxyz,[0,hiplen,0])
        lhip[0,i] = temp[0] # x,y and z coordinates 
        lhip[1,i] = temp[1]
        lhip[2,i] = temp[2]
        temp = np.dot(Rxyz,[0,-hiplen,0])
        rhip[0,i] = temp[0] # updates x,y and z coordinates
        rhip[1,i] = temp[1]
        rhip[2,i] = temp[2]
        temp = np.dot(Rxyz,[lknee[0,i],lknee[1,i],lknee[2,i]])
        lknee[0,i] = temp[0] # updates x,y and z coordinates 
        lknee[1,i] = temp[1]
        lknee[2,i] = temp[2]
        temp = np.dot(Rxyz,[rknee[0,i],rknee[1,i],rknee[2,i]])
        rknee[0,i] = temp[0] # updates x,y and z coordinates 
        rknee[1,i] = temp[1]
        rknee[2,i] = temp[2]
        temp = np.dot(Rxyz,[lankle[0,i],lankle[1,i],lankle[2,i]])
        lankle[0,i] = temp[0]# updates x,y and z coordinates 
        lankle[1,i] = temp[1]
        lankle[2,i] = temp[2]
        temp = np.dot(Rxyz,[rankle[0,i],rankle[1,i],rankle[2,i]])
        rankle[0,i] = temp[0] # updates x,y and z coordinates 
        rankle[1,i] = temp[1]
        rankle[2,i] = temp[2]
        temp = np.dot(Rxyz,[ltoe[0,i],ltoe[1,i],ltoe[2,i]])
        ltoe[0,i] = temp[0]# updates x,y and z coordinates 
        ltoe[1,i] = temp[1]
        ltoe[2,i] = temp[2]
        temp = np.dot(Rxyz,[rtoe[0,i],rtoe[1,i],rtoe[2,i]])
        rtoe[0,i] = temp[0]# updates x,y and z coordinates 
        rtoe[1,i] = temp[1]
        rtoe[2,i] = temp[2]

    # Handling upper body flexing at elbows and shoulders
    # handling flexing at the left elbow
    psi = np.linspace(0,0,nt)
    the[:round(nt/2)] = flexelbow[round(nt/2):nt]*np.pi/180
    the[round(nt/2):nt] = flexelbow[:round(nt/2)]*np.pi/180
    phi = np.linspace(0,0,nt)
    for i in range(nt):
        Rxyz =XYZConvention(psi[i],the[i],phi[i])
        temp = np.dot(Rxyz,[0,0,-lowerarmlen])
        lhand[0,i] = temp[0] #  x,y and z coordinates 
        lhand[1,i] = temp[1]+shoulderlen
        lhand[2,i] = temp[2]+(torsolen-upperarmlen)

    # handle flexing at the right elbow
    psi = np.linspace(0,0,nt)
    the = flexelbow*np.pi/180
    phi = np.linspace(0,0,nt)
    for i in range(nt):
        Rxyz =XYZConvention(psi[i],the[i],phi[i])
        temp = np.dot(Rxyz,[0,0,-lowerarmlen])
        rhand[0,i] = temp[0] #  x,y and z coordinates 
        rhand[1,i] = temp[1]-shoulderlen
        rhand[2,i] = temp[2]+(torsolen-upperarmlen)

    # handle flexing at the left shoulder
    psi = np.linspace(0,0,nt)
    the[:round(nt/2)] = flexshoulder[round(nt/2):nt]*np.pi/180 # left and right shoulder are in opposite phase
    the[round(nt/2):nt] = flexshoulder[:round(nt/2)]*np.pi/180
    phi = np.linspace(0,0,nt)
    for i in range(nt):
        Rxyz =XYZConvention(psi[i],the[i],phi[i])
        temp = np.dot(Rxyz,[0,0,-upperarmlen])
        lelbow[0,i] = temp[0] # x,y and z coordinates 
        lelbow[1,i] = temp[1]+shoulderlen
        lelbow[2,i] = temp[2]+torsolen
        temp = np.dot(Rxyz,[lhand[0,i],0,lhand[2,i]-torsolen])
        lhand[0,i] = temp[0] # updates x,y and z coordinates 
        lhand[1,i] = temp[1]+shoulderlen
        lhand[2,i] = temp[2]+torsolen

    # handle flexing at the right shoulder
    psi = np.linspace(0,0,nt)
    the = flexshoulder*np.pi/180
    phi = np.linspace(0,0,nt)
    for i in range(nt):
        Rxyz =XYZConvention(psi[i],the[i],phi[i])
        temp = np.dot(Rxyz,[0,0,-upperarmlen])
        relbow[0,i] = temp[0] # x,y and z coordinates 
        relbow[1,i] = temp[1]-shoulderlen
        relbow[2,i] = temp[2]+torsolen
        temp = np.dot(Rxyz,[rhand[0,i],0,rhand[2,i]-torsolen])
        rhand[0,i] = temp[0] # updates x,y and z coordinates 
        rhand[1,i] = temp[1]-shoulderlen
        rhand[2,i] = temp[2]+torsolen

    # Handling upper body rotation
    psi = np.linspace(0,0,nt)
    the = rotforback*np.pi/180
    phi = motthor*np.pi/180
    for i in range(nt):
        Rxyz =XYZConvention(psi[i],the[i],phi[i])
        temp = np.dot(Rxyz,[0,0,torsolen+headlen])
        head[0,i] = temp[0] # x,y and z coordinates 
        head[1,i] = temp[1]
        head[2,i] = temp[2]
        temp = np.dot(Rxyz,[0,0,torsolen])
        neck[0,i] = temp[0]# x,y and z coordinates
        neck[1,i] = temp[1]
        neck[2,i] = temp[2]
        temp = np.dot(Rxyz,[0,shoulderlen,torsolen])
        lshoulder[0,i] = temp[0]# x,y and z coordinates
        lshoulder[1,i] = temp[1]
        lshoulder[2,i] = temp[2]
        temp = np.dot(Rxyz,[0,-shoulderlen,torsolen])
        rshoulder[0,i] = temp[0]# x,y and z coordinates
        rshoulder[1,i] = temp[1]
        rshoulder[2,i] = temp[2]
        temp = np.dot(Rxyz,[lelbow[0,i],lelbow[1,i],lelbow[2,0]])
        lelbow[0,i] = temp[0]# updates x,y and z coordinates
        lelbow[1,i] = temp[1]
        lelbow[2,i] = temp[2]
        temp = np.dot(Rxyz,[relbow[0,i],relbow[1,i],relbow[2,0]])
        relbow[0,i] = temp[0]# updates x,y and z coordinates
        relbow[1,i] = temp[1]
        relbow[2,i] = temp[2]
        temp = np.dot(Rxyz,[lhand[0,i],lhand[1,i],lhand[2,i]])
        lhand[0,i] = temp[0]# updates x,y and z coordinates
        lhand[1,i] = temp[1]
        lhand[2,i] = temp[2]
        temp = np.dot(Rxyz,[rhand[0,i],rhand[1,i],rhand[2,i]])
        rhand[0,i] = temp[0]# updates x,y and z coordinates
        rhand[1,i] = temp[1]
        rhand[2,i] = temp[2]

    # The origin of the body coordinate system
    base = np.array([np.linspace(0,0,nt),np.linspace(0,0,nt),np.linspace(0,0,nt)])

    # Handling translation
    base = base+[transforback,lattrans,verttrans]
    neck = neck+[transforback,lattrans,verttrans]
    head = head+[transforback,lattrans,verttrans]
    lshoulder = lshoulder+[transforback,lattrans,verttrans]
    rshoulder = rshoulder+[transforback,lattrans,verttrans]
    lelbow = lelbow+[transforback,lattrans,verttrans]
    relbow = relbow+[transforback,lattrans,verttrans]
    lhand = lhand+[transforback,lattrans,verttrans]
    rhand = rhand+[transforback,lattrans,verttrans]
    lhip = lhip+[transforback,lattrans,verttrans]
    rhip = rhip+[transforback,lattrans,verttrans]
    lknee = lknee+[transforback,lattrans,verttrans]
    rknee = rknee+[transforback,lattrans,verttrans]
    lankle = lankle+[transforback,lattrans,verttrans]
    rankle = rankle+[transforback,lattrans,verttrans]
    ltoe = ltoe+[transforback,lattrans,verttrans]
    rtoe = rtoe+[transforback,lattrans,verttrans]

    # Animation of walking human
    if forward_motion:
        base[0,:] = base[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        neck[0,:] = neck[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        head[0,:] = head[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        lshoulder[0,:] = lshoulder[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        rshoulder[0,:] = rshoulder[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        lelbow[0,:] = lelbow[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        relbow[0,:] = relbow[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        lhand[0,:] = lhand[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        rhand[0,:] = rhand[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        lhip[0,:] = lhip[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        rhip[0,:] = rhip[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        lknee[0,:] = lknee[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        rknee[0,:] = rknee[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        lankle[0,:] = lankle[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        rankle[0,:] = rankle[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        ltoe[0,:] = ltoe[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        rtoe[0,:] = rtoe[0,:] + np.linspace(0,rlc-rlc/(nt+1),nt)
        
    segments = {}
    segments['Base'] = np.transpose(base)
    segments['Neck'] = np.transpose(neck)
    segments['Head'] = np.transpose(head)
    segments['Left Shoulder'] = np.transpose(lshoulder)
    segments['Right Shoulder'] = np.transpose(rshoulder)
    segments['Left Elbow'] = np.transpose(lelbow)
    segments['Right Elbow'] = np.transpose(relbow)
    segments['Left Hand'] = np.transpose(lhand)
    segments['Right Hand'] = np.transpose(rhand)
    segments['Left Hip'] = np.transpose(lhip)
    segments['Right Hip'] = np.transpose(rhip)
    segments['Left Knee'] = np.transpose(lknee)
    segments['Right Knee'] = np.transpose(rknee)
    segments['Left Ankle'] = np.transpose(lankle)
    segments['Right Ankle'] = np.transpose(rankle)
    segments['Left Toe'] = np.transpose(ltoe)
    segments['Right Toe'] = np.transpose(rtoe)

    for key in segments.keys():
        for i in range(1,numcyc):
            temp = np.copy(segments[key][-nt:,:])
            if forward_motion:
                temp[:,0] += rlc
            segments[key] = np.concatenate([segments[key],temp])

    # output data
    lengths = {}
    lengths['Head Length'] = headlen
    lengths['Shoulder Length'] = shoulderlen
    lengths['Torso Length'] = torsolen
    lengths['Hip Length'] = hiplen
    lengths['Upper Leg Length'] = upperleglen
    lengths['Lower Leg Length'] = lowerleglen
    lengths['Foot Length'] = footlen
    lengths['Upper Arm Length'] = upperarmlen
    lengths['Lower Arm Length'] = lowerarmlen

    return segments, lengths