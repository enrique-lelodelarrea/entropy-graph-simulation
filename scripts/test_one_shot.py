#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:00:45 2021

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
    from graphsim.ar_one_shot import ar_one_shot
#    from graphsim.aux_functions import print_elapsed_time
    level = 'info'
    log_level = logging.getLevelName(level.upper())
    logger = logs.create_logger_w_c_handler('graphsim',
                                            logger_level=log_level)
    
    # test erdos renyi one shot
    
    # undirected example
#    deg_seq = [7,8,5,1,1,2,8,10,4,2,4,5,3,6,7,3,2,7,6,1,2,9,6,1,3,4,6,3,3,3,2,4,4]
#    deg_seq = [3, 3, 3, 3]
#    deg_seq = [3, 1, 1, 1]
#    deg_seq = [4, 1, 1, 1, 1]
#    deg_seq = [2, 2, 2, 2]
#    deg_seq = [2, 2, 2, 2, 2]
#    deg_seq = [3, 3, 3, 3, 3, 3]
#    deg_seq = [3, 3, 3, 3, 3, 3, 3, 3]
#    deg_seq = [3, 2, 2, 2, 1]
#    deg_seq = [2, 1, 1, 1, 1]
#    deg_seq = [3, 2, 1, 1, 1, 1, 1]
    deg_seq = [5, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#    deg_seq = [5, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    
    # erdos renyi
    g, t = ar_one_shot(deg_seq, graph_type='undirected',
                       method='erdos-renyi',
                       prob=None, num_samples=1, max_num_tries=100000)
    print('ER: Took %s trials...' % t)
    print(g)
    
    # max entropy
    g2, t2 = ar_one_shot(deg_seq, graph_type='undirected',
                       method='max-entropy', num_samples=1, max_num_tries=100000)
    print('ME: Took %s trials...' % t2)
    print(g2)
    
    
    
     # close handlers at the end
    logs.close_handlers(logger)