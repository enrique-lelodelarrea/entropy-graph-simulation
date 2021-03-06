#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 19:43:14 2021

@author: Enrique
"""

def hours_needed(time_per_graph, num_sims_needed, num_processors=1):
    ''' Hours needed to run a simulation scheme. 
    
    Parameters
    ----------
    time_per_graph: time in seconds needed for one graph sim
    num_sims_needed: how many graphs we want to simulate
    num_processors: how many processors we have available
    
    Returns
    -------
    The time in hours needed.
    
    '''
    
    hours_needed_one_proc = num_sims_needed*time_per_graph/3600
    return hours_needed_one_proc/num_processors
    

if __name__ == '__main__':
    
    import sys
    import os
    import numpy as np
    # add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from graphsim.max_ent_binary_case import discrete_max_entr_dual
    from graphsim.undirected_graphs import graph_matrix, sample_undirected_graph, vector2graph
    
    # test 1 max entropy    
    # test dual optimization
    chesapeake = [7,8,5,1,1,2,8,10,4,2,4,5,3,6,7,3,2,
                  7,6,1,2,9,6,1,3,4,6,3,3,3,2,4,4]
    inc_mat = graph_matrix(len(chesapeake))
    print('Testing cvxpy on chesapeake degree sequence...')
    l_star_1 = discrete_max_entr_dual(A=inc_mat, b=chesapeake, method='cvxpy')
    # 41.9 ms per loop (IPython %timeit)
    print('Testing scipy on chesapeake degree sequence...')
    l_star_2 = discrete_max_entr_dual(A=inc_mat, b=chesapeake, method='root')
    # 4.24 ms per loop (IPython %timeit)
    if np.sum(np.abs(l_star_1 - l_star_1)) < 1e-3:
        print('Both methods give the same answer!')
    else:
        print('Answers varied (have to check)...')
    
    
    # test 2 sampling
    print('Sampling from a simple degree sequence...')
    # sample from a simple sequence
    d_seq = [3, 2, 2, 2, 1]
    x, _, _ = sample_undirected_graph(d_seq, rule='fixed', dual_method='cvxpy')
    print(x) 
    g = vector2graph(x, len(d_seq))
    print(g.edges)
    
    # sample from chesapeake
    print('Sampling from the Chesapeake sequence...')
    d_seq = [7,8,5,1,1,2,8,10,4,2,4,5,3,6,7,3,2,
             7,6,1,2,9,6,1,3,4,6,3,3,3,2,4,4]
    x, _, _ = sample_undirected_graph(d_seq, rule='fixed', dual_method='root')
    print(x)
    g = vector2graph(x, len(d_seq))
    print(g.edges)
    # time per graph (ipython): 7.7 seconds
    
    # time per graph root (ipython): 995 ms
    
    print('How many hours we need:')
    num_sims = 20000
    num_procs = 4
    speed = 1
    print(hours_needed(speed, num_sims, num_procs))
    