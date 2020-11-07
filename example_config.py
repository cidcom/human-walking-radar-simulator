from omegaconf import OmegaConf
from pathlib import Path
import scipy.constants

legs_only = {
    'Head' : 0,
    'Torso' : 0,
    'Left Shoulder': 0,
    'Right Shoulder': 0,
    'Left Upper Arm': 0,
    'Right Upper Arm': 0,
    'Left Lower Arm': 0,
    'Right Lower Arm': 0,
    'Left Hip': 0,
    'Right Hip': 0,
    'Left Upper Leg': 0,
    'Right Upper Leg': 0,
    'Left Lower Leg': 0,
    'Right Lower Leg': 0,
    'Left Foot': 1,
    'Right Foot': 1
}

complete_human = {
    'Head' : 1,
    'Torso' : 1,
    'Left Shoulder': 1,
    'Right Shoulder': 1,
    'Left Upper Arm': 1,
    'Right Upper Arm': 1,
    'Left Lower Arm': 1,
    'Right Lower Arm': 1,
    'Left Hip': 1,
    'Right Hip': 1,
    'Left Upper Leg': 1,
    'Right Upper Leg': 1,
    'Left Lower Leg': 1,
    'Right Lower Leg': 1,
    'Left Foot': 1,
    'Right Foot': 1
}

ExampleConfig = OmegaConf.create({
                                  "fs" : 100,
                                  "simulator" : {
                                      "forward_motion": False,
                                      "duration" : 11,
                                      "height": (1.2, 1.8),
                                      "rv": (0.2, 1.0),
                                      "radarloc": (0,10,0),
                                      "lambda_": scipy.constants.c/24e9,
                                      "rangeres": 0.01,
                                      "body_parts" : complete_human
                                  }
                                 })