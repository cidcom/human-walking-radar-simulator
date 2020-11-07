"""Radar Function Helpers

This script contains helper functions for radar simulation.

The script is based on "A global human walking model with real-time kinematic personification" paper, by R. Boulic, N.M. Thalmann, and D. Thalmann; The Visual Computer, vol.6, pp.344-358, 1990

The model is based on biomechanical experimental data.

Adopted into a Python framework by Mikolaj Czerkawski
from scripts authored by V.C. Chen and Yang Hai

"""

import numpy as np
from math import acos, asin

def XYZConvention(psi,theta,phi):
    """
    X-Y-Z convention: Roll-Pitch-Yaw convention
    
    roll: psi (rotation about x-axis)
    pitch: theta (rotation about y-axis)
    yaw: phi (rotation about z-axis)
    
    Rx = [1, 0, 0;
        0, np.cos(psi), np.sin(psi);
        0, -np.sin(psi), np.cos(psi)];
    Ry = [np.cos(theta), 0, -np.sin(theta);
        0, 1, 0;
        np.sin(theta), 0, np.cos(theta)];
    Rz = [np.cos(phi), np.sin(phi), 0;
        -np.sin(phi), np.cos(phi), 0;
        0, 0, 1];
    
    Rxyz = Rz*(Ry*Rx);
    
    By V.C. Chen
    """
    xyz= np.array([
        [np.cos(theta)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)+np.cos(psi)*np.sin(phi),-np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi)],
        [-np.cos(theta)*np.sin(phi), -np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi), np.cos(psi)*np.sin(theta)*np.sin(phi)+np.sin(psi)*np.cos(phi)],
        [np.sin(theta), -np.sin(psi)*np.cos(theta), np.cos(psi)*np.cos(theta)]
    ])
    return xyz

def rcsellipsoid(a,b,c,phi,theta):
    """
    Returns RCS of an ellipsoid with dimensions a, b, c and orientation phi, theta
    """
    return (np.pi*(a**2)*(b**2)*(c**2))/(a**2*(np.sin(theta)**2)*(np.cos(phi)**2)+b**2*(np.sin(theta)**2)*(np.sin(phi)**2)+c**2*(np.cos(theta)**2))**2

def compute_ph(aspct, position, ellipsoid, radarloc, lambda_):
    """
    Computes the value of the complex return component for an ellipsoid at a given aspect and position.
    """
    r_dist = abs(position - radarloc)
    # Distance from radar to element.
    distances = np.sqrt(r_dist[0]**2 + r_dist[1]**2 + r_dist[2]**2)
    # Calculate theta angle and phi angle (see Figure 4.30).
    A = np.array([radarloc[0]-position[0], radarloc[1]-position[1], radarloc[2]-position[2]])
    B = aspct
    A_dot_B = np.dot(A,B)
    
    A_sum_sqrt = np.sqrt(sum(A*A,1))
    B_sum_sqrt = np.sqrt(sum(B*B,1))  
    ThetaAngle = np.arccos(A_dot_B/(A_sum_sqrt*B_sum_sqrt))
    PhiAngle = asin((radarloc[1]-position[1])/np.sqrt(r_dist[0]**2+r_dist[1]**2))      
    a, b, c = ellipsoid
    # Radar cross section computation.
    rcs = rcsellipsoid(a,b,c,PhiAngle,ThetaAngle)
    amp = np.sqrt(rcs)
    # Baseband radar return. Localisation of the range bin, based on the 
    # distance, and update of the slow-time fast-time matrix.
    PHs = amp*(np.exp(-1j*4*np.pi*distances/lambda_))
    return PHs, distances