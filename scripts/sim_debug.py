#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:25:14 2021

@author: Enrique
"""

''' Script used for debugging. '''

if __name__ == '__main__':
    
    import sys
    import os
    import logging
    import numpy as np
    import time
    # add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    import graphsim.logger as logs
    from graphsim.undirected_batch_mp import sim_undirected_graphs_mp
    from graphsim.aux_functions import print_elapsed_time
    
    log_level = logging.getLevelName('WARNING')
    logger = logs.create_logger_w_c_handler('graphsim',
                                            logger_level=log_level)
    
    print('Starting the simulation (Chesapeake undirected deg sequence)')
    
    # chesapeak degree sequence
    chesapeake = [7,8,5,1,1,2,8,10,4,2,4,5,3,6,7,3,2,
                  7,6,1,2,9,6,1,3,4,6,3,3,3,2,4,4]
    
    # number of cores
    n_cores = 4
    
    # number of graphs per rule
    n_sims_rule = 60 # need 5000, but extra in case some fail due to num issues
    
    # number of graphs per batch
    n_sims_batch = 10
    
    # rule names
    rule_names = ['fixed', 'random', 'most_uniform', 'most_singular']
    
    # num batches per rule
    n_batch_rule = n_sims_rule//n_sims_batch
    
    # rules for batches
    rules_vec = np.repeat(rule_names, n_batch_rule)
    
    # number of batches
    n_batches = len(rules_vec) # 4 rules x 6 batches
    
    # debugging
    batch = 8 # start at 1
    
    print('Number of samples per rule: %s' % n_sims_rule)
    print('Sampling will be done in parallel')
    
    np.seterr(over='raise') # check for overflow
    
#    sys.exit()
    
    # run sample
    start_time = time.time()
    sim_undirected_graphs_mp(deg_seq = chesapeake,
                             num_sims_batch = n_sims_batch,
                             num_procs = 1,
                             rules = rules_vec[batch - 1],
                             prefix_file = 'ches',
                             random_seed = 12345 + 1000*(batch - 1),
                             subfolder = 'chesapeake',
                             solver = 'root',
                             num_cores = n_cores)
    end_time = time.time()
    print_elapsed_time(start_time, end_time)
    
    # close handlers at the end
    logs.close_handlers(logger)
    
    print('Done!')