"""Human Walking Simulator

This is the __init__ module for the radar human walking simulator package by Mikolaj Czerkawski

The script is based on "A global human walking model with real-time kinematic personification" paper, by R. Boulic, N.M. Thalmann, and D. Thalmann; The Visual Computer, vol.6, pp.344-358, 1990

The model is based on biomechanical experimental data.

Adopted into a Python framework by Mikolaj Czerkawski
from scripts authored by V.C. Chen and Yang Hai

"""

from .radar_helpers import *
from .simulate_radar import *
from .generate_segments import *
from .dataset_generator_tool import *