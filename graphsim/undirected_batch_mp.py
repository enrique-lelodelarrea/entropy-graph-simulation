#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:29:20 2021

@author: Enrique
"""

''' Code to simulate several undirected graphs
using multiprocessing. '''

import logging
import numpy as np
import multiprocessing as mp
import pathlib
import os

from .undirected_graphs import sample_undirected_graph

# module logger
logger = logging.getLogger(__name__)

# simulation function that will be run in parallel
def sim_batch_proc(deg_seq, num_sims, rule, prefix_file, random_seed, subfolder, solver):
    ''' 
    
    Run a batch of simulations. Each batch is run in a separate process,
    allowing for multi-processing.
    
    Parameters
    ----------
    deg_seq: integer vector
    num_sims: integer, number of simulations in the batch
    rule: string, order rule for edge simulation, currently support 4 rules
    prefix_file: string, prefix for output file
    random_seed: integer, seed for pseudo-random numbers
    subfolder: string, name of the folder where results are stored, if None, saves the results in same folder
    solver: string, determines which solver is used for solving the max-entropy problem
    
    Returns
    -------
    None
    Saves the simulation output in a file.
    '''
    
    num_fails = 0
    proc_id = mp.current_process().name
    logger.info('Starting process %s...' % proc_id)
    # set seed for cycle
    np.random.seed(random_seed)
    # alocate space
    num_nodes = len(deg_seq)
    num_edges = num_nodes*(num_nodes - 1)//2
    X_vec = np.empty([num_sims, num_edges], dtype=int)
    p_vec = np.empty(num_sims)
    w_vec = np.empty(num_sims)
    status_vec = -1*np.ones(num_sims, dtype=int)
    # simulation
    for j in range(num_sims):
        try:
            X_vec[j,:], p_vec[j], w_vec[j] = sample_undirected_graph(deg_seq, rule=rule, dual_method=solver)
            status_vec[j] = 0
        except ValueError:
            # TODO: return the random num gen for debugging
            num_fails += 1
    # create directory and file name
    outfile = '%s_proc_id_%s_%s_n_%d.npz' % (prefix_file, proc_id, rule, num_sims)
    if subfolder is not None:
        folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sims')) 
        pathlib.Path(folder, subfolder).mkdir(parents=False, exist_ok=True)
        outfile = os.path.join(folder, subfolder, outfile)
    else:
        pass # save files in the same current folder
    # save data    
    np.savez(outfile, X_vec=X_vec, p_vec=p_vec, w_vec=w_vec, status_vec=status_vec)
    logger.info('Ending process %s...' % proc_id)
    if num_fails > 0:
        logger.warning('Process %s: %d sims failed (out of %d)!' % (proc_id, num_fails, num_sims))
    return None

def sim_undirected_graphs_mp(deg_seq, num_sims_batch, num_procs, rules, prefix_file,
                             random_seed, subfolder, solver, num_cores):
    ''' 
    
    Sample random undirected graphs with prescribed degrees in parallel,
    using multiprocessing.
    
    Parameters
    ----------
    deg_seq: integer vector
    num_sims_batch: integer, number of sims per process/batch
    num_procs: integer, number of processes to be run in parallel
    rules: vector of strings, order rule for edge batch, currently support 4 rules.
           If string is given, repeat rule for all batches.
    prefix_file: string, prefix for output files
    random_seed: integer, base seed for pseudo-random numbers, will be modified for each process
    subfolder: string, name of the folder where results are stored, if None, saves the results in same folder
    solver: string, determines which solver is used for solving the max-entropy problem
    num_cores: integers, number of cores to be used in parallel
    
    Returns
    -------
    None, but it saves the simulation output in a file.
    '''  
    logger.info('Starting simulation with multi-processing...')
    # check number of cores
    max_num_cores = mp.cpu_count()
    assert num_cores <= max_num_cores
    logger.info('Will use %d cores (max is %d)' % (num_cores, max_num_cores))
    num_iters = num_procs//(num_cores) + 1 # how many times we create processes
    # start processes
    for j in range(num_iters):
        if j == num_iters-1:
            active_cores = num_procs%num_cores
        else:
            active_cores = num_cores
        # list of processes, one for each cycle
        processes = list()
        for core in range(active_cores):
            process_id = j*num_cores+core + 1 # start at one
            process_seed = random_seed + 1000*(process_id - 1) # for replicability
            # select rule for batch
            if isinstance(rules, str):
                rule = rules
            else:
                rule = rules[process_id - 1] # idx starts at zero
            new_process = mp.Process(name=str(process_id),
                                     target=sim_batch_proc,
                                     args=(deg_seq,
                                           num_sims_batch,
                                           rule,
                                           prefix_file,
                                           process_seed,
                                           subfolder,
                                           solver))
            processes.append(new_process)
        # run processes
        for p in processes:
            p.start()
        # exit the completed processes
        for p in processes:
            p.join()
    logger.info('Multi-processing is done.')
    return None