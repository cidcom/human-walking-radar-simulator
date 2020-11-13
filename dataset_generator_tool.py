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
import pandas as pd

from .simulate_radar import *
from .generate_segments import *
from .example_config import *

def generate(config_set, n_samples = 64, ddir = 'sample_dataset/', rvs = [], heights = [], gaits = [], squeeze_range = True):
    
    if type(config_set) == list:
        num_configs = len(config_set)
        
        # get equal distribution of configs
        config_idxs = []
        for another_config in range(num_configs-1):
            config_idxs += [another_config for i in range(int(n_samples/num_configs))]
        config_idxs += [num_configs-1 for i in range(n_samples - len(config_idxs))]
        
        # basic parameters from the first config
        config = config_set[0]
        
        # save config
        for cf_idx in range(len(config_set)):
            cf = config_set[cf_idx]
            conf_file = open(ddir + "dataset_configuration_" + str(cf_idx) + ".txt","w")
            conf_file.write(cf.pretty())
            conf_file.close()
    else:
        config = config_set
        conf_file = open(ddir + "dataset_configuration.txt","w")
        conf_file.write(config.pretty())
        conf_file.close()
    
    forward_motion = config['forward_motion']
    nt = config.simulator['nt']
    numcyc = config.simulator['numcyc']
    radarloc = config.simulator['radarloc']
    rangeres = config.simulator['rangeres']
    lambda_ = config.simulator['lambda_']
    fs = config['fs']
    
    sim_config = config['sim_config']
    
    if heights == []:
        heights = np.random.uniform(config.simulator['height'][0],config.simulator['height'][1],n_samples)
    if rvs == []:
        rvs = np.random.uniform(config.simulator['rv'][0],config.simulator['rv'][1], n_samples)
    if gaits == []:
        gaits = ['' for i in range(n_samples)]
    
    # create directory
    if not os.path.exists(ddir):
        os.mkdir(ddir)
    
    # pandas dataframe
    df = pd.DataFrame(None, columns = ['sample_idx','rv', 'height', 'config','radarloc','fs','forward_motion'])
    
    # generate in loop
    for sample_idx in tqdm(range(n_samples)):
        height = heights[sample_idx]
        rv = rvs[sample_idx]
        gait = gaits[sample_idx]
        seg,segl  = generate_segments(forward_motion = forward_motion,
                                      height = height,
                                      rv = rv,
                                      fs = fs,
                                      gait = gait,
                                      duration = config.simulator.duration,
                                      radarloc = radarloc)
        if type(config_set) == list:
            cf_idx = config_idxs[sample_idx]
            sim_config = config_set[cf_idx]['sim_config']
        else:
            cf_idx = 0

        mat = simulate_radar(seg, segl, lambda_ = lambda_, rangeres = rangeres, radarloc = radarloc, config = sim_config)
        
        if squeeze_range:
            mat = sum(mat) 
            mu_i = np.mean(abs(mat))
            mat -= mu_i
            std_i = np.std(abs(mat))
            mat /= std_i

        with open(ddir + '/sample' + str(sample_idx+1) + '.npy', 'wb') as f:
            np.save(f, mat)
                   
        df = df.append(pd.DataFrame([[sample_idx+1, rv, height, cf_idx, radarloc, fs, forward_motion]],
                                    columns=['sample_idx','rv', 'height','config','radarloc','fs','forward_motion']))
    df.to_pickle(ddir + "dataframe.pkl")
    

if __name__ == '__main__':

    config = ExampleConfig
    
    n_samples = 1024

    generate(config, n_samples)