"""Radar Return Simulator

This script reads the human kinematics data contained in the segment dictionary and simulates the radar range-time map for that signal.

The script is based on "A global human walking model with real-time kinematic personification" paper, by R. Boulic, N.M. Thalmann, and D. Thalmann; The Visual Computer, vol.6, pp.344-358, 1990

The model is based on biomechanical experimental data.

Adopted into a Python framework by Mikolaj Czerkawski
from scripts authored by V.C. Chen and Yang Hai

"""

import numpy as np
from math import acos, asin

from .radar_helpers import *

def simulate_radar(segment, seglength, lambda_, rangeres, radarloc, config = None):
    """Simulates the radar range-time map based on the input kinematics data.

    Parameters
    ----------
    segment : dict
        A dictionary containing the kinematic traces of the reference body points
        
    seglength : dict
        A dictionary containing lengths of the body parts
        
    lambda_ : float
        Simulated carrier wavelength in meters
    
    rangeres: float
        Simulated range resolution in meters
    
    radarloc: tuple
        Location of the radar receiver (x, y, z)
    
    config : OmegaConf
        Configuration object

    Returns
    -------
    numpy array
        a complex range-time map
    """

    headlen = seglength['Head Length']
    shoulderlen = seglength['Shoulder Length']
    torsolen = seglength['Torso Length']
    hiplen = seglength['Hip Length']
    upperleglen = seglength['Upper Leg Length']
    lowerleglen = seglength['Lower Leg Length']
    footlen = seglength['Foot Length']
    upperarmlen = seglength['Upper Leg Length']
    lowerarmlen = seglength['Lower Arm Length']
    
    base = segment['Base']
    neck = segment['Neck']
    head = segment['Head']
    lshoulder = segment['Left Shoulder']
    rshoulder = segment['Right Shoulder']
    lelbow = segment['Left Elbow']
    relbow = segment['Right Elbow']
    lhand = segment['Left Hand']
    rhand = segment['Right Hand']
    lhip = segment['Left Hip']
    rhip = segment['Right Hip']
    lknee = segment['Left Knee']
    rknee = segment['Right Knee']
    lankle = segment['Left Ankle']
    rankle = segment['Right Ankle']
    ltoe = segment['Left Toe']
    rtoe = segment['Right Toe']
    
    if config == None:
        config = {}
        c_keys = ['Head',
                  'Torso',
                  'Left Shoulder',
                  'Right Shoulder',
                  'Left Upper Arm',
                  'Right Upper Arm',
                  'Left Lower Arm',
                  'Right Lower Arm',
                  'Left Hip',
                  'Right Hip',
                  'Left Upper Leg',
                  'Right Upper Leg',
                  'Left Lower Leg',
                  'Right Lower Leg',
                  'Left Foot',
                  'Right Foot']
        for key in c_keys:
            config[key] = True

    # Computation of the number of range bins based on the selected range
    # resolution.
    nr = round(2*np.sqrt(radarloc[0]**2+radarloc[1]**2+radarloc[2]**2)/rangeres)
    # Number of slow-time pulses.
    numpl = base.shape[0]

    # Allocation of the slow-time fast-time matrix.
    data = np.zeros([nr,numpl], dtype = 'complex')

    for k in range(numpl):
        # Radar returns from the head.
        if config['Head']:
            aspct = head[k,:]-neck[k,:]
            ph, distances = compute_ph(aspct, head[k,:], ellipsoid = (0.1, 0.1, headlen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph
        
        # Radar returns from torso.
        if config['Torso']:
            torso = (neck[k,:]+base[k,:])/2
            aspct = neck[k,:]-base[k,:]        
            ph, distances = compute_ph(aspct, torso, ellipsoid = (0.15, 0.15, torsolen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph
        
        # Radar returns from left shoulder.
        if config['Left Shoulder']:
            aspct = lshoulder[k,:]-neck[k,:]        
            ph, distances = compute_ph(aspct, lshoulder[k,:], ellipsoid = (0.06, 0.06, shoulderlen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph
        
        # Radar returns from right shoulder.
        if config['Right Shoulder']:
            aspct = rshoulder[k,:]-neck[k,:]        
            ph, distances = compute_ph(aspct, rshoulder[k,:], ellipsoid = (0.06, 0.06, shoulderlen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph

        # Radar returns from left upper-arm.
        if config['Left Upper Arm']:
            lupperarm = (lshoulder[k,:]+lelbow[k,:])/2
            aspct = lshoulder[k,:]-lelbow[k,:]        
            ph, distances = compute_ph(aspct, lupperarm, ellipsoid = (0.06, 0.06, upperarmlen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph
        
        # Radar returns from right upper-arm.
        if config['Right Upper Arm']:
            rupperarm = (rshoulder[k,:]+relbow[k,:])/2
            aspct = rshoulder[k,:]-relbow[k,:]        
            ph, distances = compute_ph(aspct, rupperarm, ellipsoid = (0.06, 0.06, upperarmlen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph
        
        # Radar returns from left lower-arm.
        if config['Left Lower Arm']:
            aspct = lelbow[k,:]-lhand[k,:]        
            ph, distances = compute_ph(aspct, lhand[k,:], ellipsoid = (0.05, 0.05, lowerarmlen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph
        
        # Radar returns from right lower-arm.
        if config['Right Lower Arm']:
            aspct = relbow[k,:]-rhand[k,:]        
            ph, distances = compute_ph(aspct, rhand[k,:], ellipsoid = (0.05, 0.05, lowerarmlen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph
        
        # Radar returns from left hip.
        if config['Left Hip']:
            aspct = lhip[k,:]-base[k,:]        
            ph, distances = compute_ph(aspct, lhip[k,:], ellipsoid = (0.07, 0.07, hiplen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph   
        
        # Radar returns from right hip.
        if config['Right Hip']:
            aspct = rhip[k,:]-base[k,:]        
            ph, distances = compute_ph(aspct, rhip[k,:], ellipsoid = (0.07, 0.07, hiplen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph   
        
        # Radar returns from left upper-leg.
        if config['Left Upper Leg']:
            lupperleg = (lhip[k,:]+lknee[k,:])/2
            aspct = lknee[k,:]-lhip[k,:]        
            ph, distances = compute_ph(aspct, lupperleg, ellipsoid = (0.07, 0.07, upperleglen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph 
        
        # Radar returns from right upper-leg.
        if config['Right Upper Leg']:
            rupperleg = (rhip[k,:]+rknee[k,:])/2
            aspct = rknee[k,:]-rhip[k,:]        
            ph, distances = compute_ph(aspct, rupperleg, ellipsoid = (0.07, 0.07, upperleglen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph
        
        # Radar returns from left lower-leg.
        if config['Left Lower Leg']:
            llowerleg = (lankle[k,:]+lknee[k,:])/2
            aspct = lankle[k,:]-lknee[k,:]        
            ph, distances = compute_ph(aspct, llowerleg, ellipsoid = (0.06, 0.06, lowerleglen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph
        
        # Radar returns from right lower-leg.
        if config['Right Lower Leg']:
            rlowerleg = (rankle[k,:]+rknee[k,:])/2
            aspct = rankle[k,:]-rknee[k,:]        
            ph, distances = compute_ph(aspct, rlowerleg, ellipsoid = (0.06, 0.06, lowerleglen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph
        
        # Radar returns from left foot.
        if  config['Left Foot']:
            aspct = lankle[k,:]-ltoe[k,:]        
            ph, distances = compute_ph(aspct, ltoe[k,:], ellipsoid = (0.05, 0.05, footlen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph
             
        # Radar returns from right foot.
        if config['Right Foot']:
            aspct = rankle[k,:]-rtoe[k,:]    
            ph, distances = compute_ph(aspct, rtoe[k,:], ellipsoid = (0.05, 0.05, footlen/2), radarloc = radarloc, lambda_ = lambda_)
            r_index = int(np.floor(distances/rangeres))
            data[r_index,k] += ph

    return data

