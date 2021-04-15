#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:20:29 2021

@author: Enrique
"""

if __name__ == '__main__':
    
    import sys
    import os
    # add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from graphsim.bipartite_graphs import sample_bipartite_graph, vector_to_matrix 
    
    # test 1 from small
    print('Sampling from a simple degree sequence...')
    # sample from a simple sequence
    r = [1, 1, 2]
    c = [1, 1, 2]
    x1, _, _ = sample_bipartite_graph(r, c, rule='fixed', dual_method='root')
    g1 = vector_to_matrix(x1, len(r), len(c))
    print(g1)
    
    # test 2 from interbank1
    print('Sampling from a Interbank1...')
    # sample from a simple sequence
    r = [6,6,6,5,5,5,5,4,4,3,2]
    c = [6,6,6,5,5,5,5,4,4,3,2]
    m = len(r)
    n = len(c)
    x2, _, _ = sample_bipartite_graph(r, c, rule='fixed', dual_method='root')
    g2 = vector_to_matrix(x2, len(r), len(c))
    print(g2)
    
    # test 3 from chesapeake
    print('Sampling from a Chesapeake...')
    # sample from a simple sequence
    r = [7, 8, 5, 1, 1, 1, 5, 7, 1, 0, 1, 2, 0, 5, 6, 2, 0, 6, 2, 0, 1, 6, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    c = [0, 0, 0, 0, 0, 1, 3, 3, 3, 2, 3, 3, 3, 1, 1, 1, 2, 1, 6, 1, 1, 3, 3, 1, 3, 4, 5, 3, 3, 3, 1, 4, 4]
    m = len(r)
    n = len(c)
    x3, _, _ = sample_bipartite_graph(r, c, rule='fixed', dual_method='root')
    g3 = vector_to_matrix(x3, len(r), len(c))
    print(g3)
    
    print('Done!')
    
    