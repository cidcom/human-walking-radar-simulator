"""Dataset Generator

The following script is capable of generating a human walking radar return dataset.

This file can also be imported as a module and contains the following
functions:

    * generate - generates a dataset when called based on the supplied configuration
"""

import numpy as np
import os
import multiprocessing
from tqdm import tqdm
import datetime
import pandas as pd

from .simulate_radar import *
from .generate_segments import *
from .example_config import *

def _single_config_generation(n_samples, offset, heights, rvs, gaits, config_set, cf_idx, ddir = '.', squeeze_range = True):
    df = pd.DataFrame(None, columns = ['sample_idx','rv', 'height', 'config','radarloc','fs','forward_motion'])
    
    config = config_set[cf_idx]
    fs = config.fs
    sim_config = config.simulator
    forward_motion = sim_config.forward_motion
    duration = sim_config.duration
    radarloc = sim_config.radarloc
    lambda_ = sim_config.lambda_
    rangeres = sim_config.rangeres
    
    
    if cf_idx == 0:
        iterable = tqdm(range(offset,offset+n_samples), desc = 'Conf1-Progress')
    else:
        iterable = range(offset,offset+n_samples)
    
    for sample_idx in iterable:
        if os.path.isfile(ddir + '/sample' + str(sample_idx+1) + '.npy'):
            continue
        
        height = heights[sample_idx]
        rv = rvs[sample_idx]
        gait = gaits[sample_idx]
        seg,segl  = generate_segments(forward_motion = forward_motion,
                                      height = height,
                                      rv = rv,
                                      fs = fs,
                                      gait = gait,
                                      duration = duration,
                                      radarloc = radarloc)

        mat = simulate_radar(seg, segl, lambda_ = lambda_, rangeres = rangeres, radarloc = radarloc, config = sim_config.body_parts)
        
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
    return df

def generate(config_set, n_samples = 64, ddir = 'sample_dataset/', rvs = [], heights = [], gaits = [], squeeze_range = True):        
    # create directory
    if not os.path.exists(ddir):
        os.mkdir(ddir)
        
    if type(config_set) == list:
        num_configs = len(config_set)
        
        config_counts = []
        config_offsets = []
        total_counts = 0
        for another_config in range(num_configs-1):
            config_offsets += [total_counts]
            new_count = int(n_samples/num_configs)
            config_counts += [new_count]
            total_counts += new_count
        config_counts += [n_samples - total_counts]
        config_offsets += [total_counts]
        
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
    
    if heights == []:
        heights = np.random.uniform(config.simulator['height'][0],config.simulator['height'][1],n_samples)
    if rvs == []:
        rvs = np.random.uniform(config.simulator['rv'][0],config.simulator['rv'][1], n_samples)
    if gaits == []:
        gaits = ['' for i in range(n_samples)]
    
    # pandas dataframe
    df = pd.DataFrame(None, columns = ['sample_idx','rv', 'height', 'config','radarloc','fs','forward_motion'])
    
    # generate in pool
    # (n_samples, offset, heights, rvs, gaits, forward_motion, fs, duration, radarloc, config_set, cf_idx)
    pool = multiprocessing.Pool()
    pool_dfs = pool.starmap(_single_config_generation, zip(config_counts,
                                                           config_offsets,
                                                           [heights for cf_idx in range(num_configs)],
                                                           [rvs for cf_idx in range(num_configs)],
                                                           [gaits for cf_idx in range(num_configs)],
                                                           [config_set for cf_idx in range(num_configs)],
                                                           [cf_idx for cf_idx in range(num_configs)],
                                                           [ddir for cf_idx in range(num_configs)],
                                                           [squeeze_range for cf_idx in range(num_configs)]
                                                           ))
    pool.close()
    
    for pool_df in pool_dfs:
        df = df.append(pool_df)
    df.to_pickle(ddir + "dataframe.pkl")
    

if __name__ == '__main__':

    config = ExampleConfig
    
    n_samples = 1024

    generate(config, n_samples)