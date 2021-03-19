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
    # add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    import graphsim.logger as logs
    from graphsim.undirected_batch_mp import sim_undirected_graphs_mp
    
    log_level = logging.getLevelName('WARNING')
    logger = logs.create_logger_w_c_handler('graphsim',
                                            logger_level=log_level)
    
    
    # simulation setting
    degrees = [3, 2, 2, 2, 1]
    
    # run sample
    sim_undirected_graphs_mp(deg_seq = degrees,
                             num_sims_batch = 10,
                             num_procs = 1,
                             rule = 'fixed',
                             prefix_file = 't_sim',
                             random_seed = 123,
                             run_name = 'test',
                             solver = 'cvxpy',
                             all_cores = False)
    
    # close handlers at the end
    logs.close_handlers(logger)
    
    
    
    