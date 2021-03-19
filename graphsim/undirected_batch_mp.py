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
def sim_batch_proc(proc_id, deg_seq, num_sims, rule, prefix_file, random_seed, subfolder, solver):
    ''' 
    
    Run a batch of simulations. Each batch is run in a separate process,
    allowing for multi-processing.
    
    Parameters
    ----------
    proc_id: integer, number id for batch, affects also the random seed
    deg_seq: integer vector
    num_sims: integer, number of simulations in the batch
    rule: string, order rule for edge simulation, currently support 4 rules
    prefix_file: string, prefix for output file
    random_seed: integer, base seed for pseudo-random numbers, modified also by proc_id
    subfolder: string, name of the folder where results are stored, if None, saves the results in same folder
    solver: string, determines which solver is used for solving the max-entropy problem
    
    Returns
    -------
    None, but it saves the simulation output in a file.
    '''
    
    name = mp.current_process().name
    logger.info('Starting process %s...' % name)
    # set seed for cycle
    np.random.seed(random_seed + 1000*proc_id)
    # alocate space
    num_nodes = len(deg_seq)
    num_edges = num_nodes*(num_nodes - 1)//2
    X_vec = np.empty([num_sims, num_edges], dtype=int)
    p_vec = np.empty(num_sims)
    w_vec = np.empty(num_sims)
    status_vec = -1*np.ones(num_sims, dtype=int)
    # simulation
    for j in range(num_sims):
        X_vec[j,:], p_vec[j], w_vec[j] = sample_undirected_graph(deg_seq, rule=rule, dual_method=solver)
        status_vec[j] = 0
    # create directory and file name
    outfile = '%s_%s_n_%d_batch_%d.npz' % (prefix_file, rule, num_sims, proc_id)
    if subfolder is not None:
        pathlib.Path('../sims/', subfolder).mkdir(parents=False, exist_ok=True)
        folder = os.path.join('../sims', subfolder)
        outfile = os.path.join(folder, outfile)
    # save data    
    np.savez(outfile, X_vec=X_vec, p_vec=p_vec, w_vec=w_vec, status_vec=status_vec)
    logger.info('Ending process %s...' % name)
    return None

def sim_undirected_graphs_mp(deg_seq, num_sims_batch, num_procs, rule, prefix_file,
                             random_seed, run_name, solver, all_cores):
    ''' 
    
    Sample random undirected graphs with prescribed degrees in parallel,
    using multiprocessing.
    
    Parameters
    ----------
    deg_seq: integer vector
    num_sims_batch: integer, number of sims per process/batch
    num_procs: integer, number of processes to be run in parallel
    rule: string, order rule for edge simulation, currently support 4 rules
    prefix_file: string, prefix for output files
    random_seed: integer, base seed for pseudo-random numbers, will be modified for each process
    run_name: string, name of the sim run and the folder where results are stored, if None, saves the results in same folder
    solver: string, determines which solver is used for solving the max-entropy problem
    all_cores: boolean, use all cores if True, leave one unused if False
    
    Returns
    -------
    None, but it saves the simulation output in a file.
    '''  
    logger.info('Starting simulation with multi-processing...')
    # determine number of cores
    if all_cores:
        num_cores = mp.cpu_count() 
    else:
        num_cores = mp.cpu_count() - 1
    logger.info('Will use %d cores' % num_cores)
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
            new_process = mp.Process(name="process_%s"%str(process_id),
                                     target=sim_batch_proc,
                                     args=(process_id,
                                           deg_seq,
                                           num_sims_batch,
                                           rule,
                                           prefix_file,
                                           random_seed,
                                           run_name,
                                           solver))
            processes.append(new_process)
        # run processes
        for p in processes:
            p.start()
        # exit the comprelted processes
        for p in processes:
            p.join()
    logger.info('Multi-processing is done.')
    return None