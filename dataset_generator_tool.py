"""Dataset Generator

The following script is capable of generating a human walking radar return dataset.

This file can also be imported as a module and contains the following
functions:

    * generate - generates a dataset when called based on the supplied configuration
"""

import numpy as np
import os
from tqdm import tqdm
import datetime

from .simulate_radar import *
from .generate_segments import *

from ..conf import *

def generate(config, n_samples = 64, ddir = 'sample_dataset/', squeeze_range = True):
    
    forward_motion = config['forward_motion']
    nt = config.simulator['nt']
    numcyc = config.simulator['numcyc']
    radarloc = config.simulator['radarloc']
    rangeres = config.simulator['rangeres']
    lambda_ = config.simulator['lambda_']
    fs = config['target_sampling_rate']
    
    sim_config = config['sim_config']
    
    heights = np.random.uniform(config.simulator['height'][0],config.simulator['height'][1],n_samples)
    rvs = np.random.uniform(config.simulator['rv'][0],config.simulator['rv'][1], n_samples)
    
    # create directory
    if not os.path.exists(ddir):
        os.mkdir(ddir)
        
    # save config
    conf_file = open(ddir + "dataset_configuration.txt","w")
    conf_file.write(config.pretty())
    conf_file.close()
    
    # generate in loop
    for sample_idx in tqdm(range(n_samples)):
        height = heights[sample_idx]
        rv = rvs[sample_idx]

        seg,segl,T  = generate_segments(forward_motion = forward_motion,
                                        height = height,
                                        rv = rv,
                                        fs = fs,
                                        duration = config.simulator.duration,
                                        radarloc = radarloc)

        mat = simulate_radar(seg, segl, T, lambda_ = lambda_, rangeres = rangeres, radarloc = radarloc, config = sim_config)
        
        if squeeze_range:
            mat = sum(mat) 
            mu_i = np.mean(abs(mat))
            mat -= mu_i
            std_i = np.std(abs(mat))
            mat /= std_i

        with open(ddir + '/sample' + str(sample_idx+1) + '.npy', 'wb') as f:
            np.save(f, mat)
    

if __name__ == '__main__':

    config = DatasetConfig
    
    n_samples = 1024

    generate(config, n_samples)