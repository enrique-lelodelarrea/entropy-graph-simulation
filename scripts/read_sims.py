#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:39:22 2021

@author: Enrique
"""

import numpy as np
import os

def read_sims_file(folder, prefix_file, rule, num_sims, file_indices, filter_success=True, num_sims_ret=None):
    ''' 
    
    Read simulation data from a sequence of files.
    
    Parameters
    ----------
    folder: string, folder path for files
    prefix_file: string, prefix name for files
    rule: string, simulation scheme, "fixed", "most_uniform", "random"
    num_sims: integer or integer array, number of simulation per file
    file_indices: list, indices of files to read
    filter_success: boolean, filter failed simulations
    num_sims_ret: integer, at most how many simulations will return, if None, return all
    
    Returns
    -------
    X: numeric array, 3d matrix with all simulated graphs (only succesful trials)
    w: numeric array, simulated weights
        
    '''
    
    num_files = len(file_indices)
    if isinstance(num_sims, int):
        n = num_files * num_sims
        num_sims = np.ones(num_files, dtype=int)*num_sims # create an array
    else:
        n = np.sum(num_sims)
    w = np.empty(n)
    status = np.zeros(n, dtype=int) # zero means success
    
    first_idx = 0
    for i, idx in enumerate(file_indices):
        last_idx = first_idx + num_sims[i]
        f_name = '%s_proc_id_%s_%s_n_%d.npz' % (prefix_file, idx, rule, num_sims[i])
        f = os.path.join(folder, f_name)
        print('reading ' + f)
        data_batch = np.load(f)
        if i == 0:
            X_00 = data_batch['X_vec'][0]
            m = len(X_00)
            X = np.empty([n,m], dtype=int)
        X[first_idx:last_idx,:] = data_batch['X_vec']
        w[first_idx:last_idx] = data_batch['w_vec']
        if filter_success:
            status[first_idx:last_idx] = data_batch['status_vec']
        first_idx = last_idx
    success_sims = np.where(status == 0)
    if num_sims_ret is None:
        return X[success_sims], w[success_sims]
    else:
        return X[success_sims][:num_sims_ret], w[success_sims][:num_sims_ret]

def weights_scaling_stats(w, scale=1):
    '''
    
    Scale simulation weights and print some summary stats.
    
    Parameters
    ----------
    w: numeric vector, importance sampling weights (up to mult constant)
    scale: numeric, positive factor by which weights will be scaled
        
    Returns
    -------
    w_scale: numeric vector, scaled importance sampling weights
    w_dict: dictionary, summary statistics
        
    '''
    w_dict = {}
    print('scaling weigths by ' + str(scale))
    w_scaled = 1. * w / scale
    mean = np.mean(w_scaled)
    w_dict['mean'] = mean
    minimum = np.min(w_scaled)
    w_dict['min'] = minimum
    maximum = np.max(w_scaled)
    w_dict['max'] = maximum
    median = np.median(w_scaled)
    w_dict['median'] = median
    sd = np.std(w_scaled)
    w_dict['sd'] = sd
    diagnosis1 = maximum / median
    w_dict['max/median'] = diagnosis1
    # other measure (less than 0.01 is good)
    diagnosis2 = maximum / np.sum(w_scaled)
    w_dict['max/sum'] = diagnosis2
    return w_scaled, w_dict

if __name__ == '__main__':
    
    # test the reading
    
    folder = '../sims/chesapeake'
    prefix_file = 'ches'
    rule = 'fixed'
    num_sims = 1000
    file_indices = [1, 2, 3, 4, 5, 6]
    num_files = 6
    filter_success = True
    num_sims_ret = None
    
    X, w = read_sims_file(folder, prefix_file, rule, num_sims, file_indices, filter_success=True, num_sims_ret=None)
    
    print(np.mean(w)) # use for scaling