#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:50:44 2021

@author: Enrique
"""

''' Compare acceptance probability of one-shot algorithm.
Erdos-Renyi vs Max Entropy.
'''

def estimate_accept_prob(sample):
    '''
    Estimate the probability of success of a geometric r.v. (w.o. mass at 0).
    '''
    n = len(sample)
    assert n > 1
    p_hat = (n - 1)/(sum(sample) - 1)
    return p_hat

def expected_trials(num_nodes, num_graphs):
    return 2**(num_nodes*(num_nodes - 1)/2)/num_graphs

if __name__ == '__main__':
    
    import sys
    import os
    import logging
    import time
    import numpy as np
    # add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    import graphsim.logger as logs
    from graphsim.ar_one_shot import ar_one_shot
    from graphsim.aux_functions import print_elapsed_time
    level = 'warning'
    log_level = logging.getLevelName(level.upper())
    logger = logs.create_logger_w_c_handler('graphsim',
                                            logger_level=log_level)
    
#    # only valid when ER uses p=0.5.
#    
#    # number of graphs is taken from sequence A002829 in Sloane's
#    print('Exact expected number of trials for 3-regular graphs:')
#    print('n = 4, value = %f' % expected_trials(4, 1))
#    print('n = 6, value = %f' % expected_trials(6, 70))
#    print('n = 8, value = %f' % expected_trials(8, 19355))
#    print('n = 10, value = %f' % expected_trials(10, 11180820))
#    
#    # for stars
#    print('Exact expected number of trials for stars:')
#    print('n = 4, value = %f' % expected_trials(4, 1))
#    print('n = 6, value = %f' % expected_trials(5, 1))
#    print('n = 8, value = %f' % expected_trials(6, 1))
    
    
    sequences = [
#                 [3, 3, 3, 3],
#                 [3, 1, 1, 1],
#                 [4, 1, 1, 1, 1],
#                 [2, 2, 2, 2, 2],
#                 [3, 2, 2, 2, 1],
#                 [2, 1, 1, 1, 1],
#                 [3, 2, 1, 1, 1, 1, 1],
#                 [3, 3, 3, 3, 3, 3, 3, 3], # takes a couple of minutes
#                 [5, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1], # ER times out
                 [4, 3, 1, 1, 1, 1, 1, 1, 1]
                 ]
    
#    deg_seq = [7,8,5,1,1,2,8,10,4,2,4,5,3,6,7,3,2,7,6,1,2,9,6,1,3,4,6,3,3,3,2,4,4]
    
    # set seed
    num_graphs = 1000
    max_trials = 1000000 # max number trials per iteration of AR
    
    p_hat_er = list()
    x_hat_er = list()
    p_hat_max_ent = list()
    x_hat_max_ent = list()
    
    improve = list()
    
    verbose = True
    
    print('Sample size per degree sequence: %d' % num_graphs)
    
    st = time.time()
    for i, deg_seq in enumerate(sequences):
        np.random.seed(12345) # the order of sequences wont matter
        print('\nDegree sequence:')
        print(deg_seq)
        print('Erdos-Renyi...')
        st = time.time()
        # erdos renyi
        try:
            _, trials = ar_one_shot(deg_seq, graph_type='undirected',
                                    method='erdos-renyi',
                                    prob=None, num_samples=num_graphs,
                                    max_num_tries=max_trials,
                                    print_progress=verbose)
            p_hat_er.append(estimate_accept_prob(trials))
            x_hat_er.append(sum(trials)/num_graphs)
            print('Erdos-Renyi accept prob: %.4f' % p_hat_er[i])
            print('Erdos-Renyi expected trials: %.4f' % x_hat_er[i])
        except SystemExit:
            logger.error('Erdos-Renyi timed out!')
            p_hat_er.append(0)
            x_hat_er.append(0)
        et = time.time()
        print_elapsed_time(st, et)
        # max entropy
        print('Maximum entropy...')
        st = time.time()
        _, trials = ar_one_shot(deg_seq, graph_type='undirected',
                           method='max-entropy', num_samples=num_graphs,
                           max_num_tries=max_trials,
                           print_progress=verbose)
        p_hat_max_ent.append(estimate_accept_prob(trials))
        x_hat_max_ent.append(sum(trials)/num_graphs)
        print('Max-entropy accept prob: %.4f' % p_hat_max_ent[i])
        print('Max-entropy expected trials: %.4f' % x_hat_max_ent[i])
        et = time.time()
        print_elapsed_time(st, et)
        try:
            temp = p_hat_max_ent[i]/p_hat_er[i] - 1
            improve.append(temp)
            print('Relative improvement: %.2f%%' % (100*improve[i]))
        except:
            pass
     # close handlers at the end
    logs.close_handlers(logger)
    print('Done!')
    
    try:
        print('Ratio')
        p_hat_er = np.array(p_hat_er)
        p_hat_max_ent = np.array(p_hat_max_ent)
        print(p_hat_max_ent/p_hat_er - 1)
    except:
        pass