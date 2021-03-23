#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:33:01 2021

@author: Enrique
"""

''' Test the sampling of undirected graphs using multiprocessing. '''

if __name__ == '__main__':
    
    import sys
    import os
    import logging
    import time
    # add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    import graphsim.logger as logs
    from graphsim.undirected_batch_mp import sim_undirected_graphs_mp
    from graphsim.aux_functions import print_elapsed_time
    
    log_level = logging.getLevelName('WARNING')
    logger = logs.create_logger_w_c_handler('graphsim',
                                            logger_level=log_level)
    
    
    # simulation setting
    degrees = [3, 2, 2, 2, 1]
    
    print('Running samples in parallel...')
    
    # run simulation
    start_time = time.time()
    sim_undirected_graphs_mp(deg_seq = degrees,
                             num_sims_batch = 10,
                             num_procs = 4,
                             rules = ['fixed', 'random', 'most_uniform', 'most_singular'],
                             prefix_file = 'testing',
                             random_seed = 12345,
                             subfolder = 'test_mp',
                             solver = 'cvxpy',
                             num_cores = 3)
    end_time = time.time()
    print_elapsed_time(start_time, end_time)
    
    # close handlers at the end
    logs.close_handlers(logger)
    
    print('Done!')
    
    
    
    