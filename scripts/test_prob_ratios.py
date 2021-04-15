#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:25:22 2021

@author: Enrique
"""

if __name__ == '__main__':
    
    import sys
    import os
    import logging
#    import time
    # add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    import graphsim.logger as logs
    from graphsim.probs_er_and_me import ratio_ME_to_ER, acceptance_prob_ME
#    from graphsim.aux_functions import print_elapsed_time
    level = 'info'
    log_level = logging.getLevelName(level.upper())
    logger = logs.create_logger_w_c_handler('graphsim',
                                            logger_level=log_level)
    from test_one_shot import connected_instance
    
    
    # undirected example
    deg_seqs = [[7,8,5,1,1,2,8,10,4,2,4,5,3,6,7,3,2,7,6,1,2,9,6,1,3,4,6,3,3,3,2,4,4],
                [3, 3, 3, 3],
                [3, 1, 1, 1],
                [4, 1, 1, 1, 1],
                [2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 3, 3, 3],
                [3, 2, 2, 2, 1],
                [2, 1, 1, 1, 1],
                [3, 2, 1, 1, 1, 1, 1],
                [4, 3, 1, 1, 1, 1, 1, 1, 1],
                [5, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [6, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                connected_instance(28, 1)]
    
    for deg_seq in deg_seqs:
        print()
        print('Degree sequence is %s' % deg_seq)
        print('Number of nodes is %s' % len(deg_seq))
    
        ratio = ratio_ME_to_ER(deg_seq)
        print('Ratio ME to ER is %g' % ratio)
    
    
     # close handlers at the end
    logs.close_handlers(logger)
    
    # for the connected instance we know the cardinality of the set
    d_seq = connected_instance(28, 1)
    size = len(d_seq)
    card = (size - 2)*(size - 3) + 1
    acc_prob = acceptance_prob_ME(d_seq, card)
    print('Degree sequence is %s' % d_seq)
    print('Number of nodes is %s' % size)
    print('Acc prob with ME: %g' % acc_prob)
    
    print('Done!')