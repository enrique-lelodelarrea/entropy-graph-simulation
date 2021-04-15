#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:38:22 2021

@author: Enrique
"""

if __name__ == '__main__':
    
    import sys
    import os
    # add parent directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from graphsim.bipartite_graphs import sample_bipartite_graph 
    from graphsim.undirected_graphs import sample_undirected_graph
    
    seqs = {
            'interbank1':
                {'type': 'bipartite',
                 'r':[6,6,6,5,5,5,5,4,4,3,2],
                 'c':[6,6,6,5,5,5,5,4,4,3,2]},
            'interbank2':
                {'type': 'bipartite',
                 'r':[9,9,9,9,9,9,8,8,8,7,6],
                 'c':[9,9,9,9,9,9,8,8,8,7,6]},
            'chesapeake':
                {'type': 'bipartite',
                 'r':[7, 8, 5, 1, 1, 1, 5, 7, 1, 0, 1, 2, 0, 5, 6, 2, 0, 6, 2, 0, 1, 6, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                 'c':[0, 0, 0, 0, 0, 1, 3, 3, 3, 2, 3, 3, 3, 1, 1, 1, 2, 1, 6, 1, 1, 3, 3, 1, 3, 4, 5, 3, 3, 3, 1, 4, 4]},
            'chesapeake-und':
                {'type': 'undirected',
                 'd':[7,8,5,1,1,2,8,10,4,2,4,5,3,6,7,3,2,7,6,1,2,9,6,1,3,4,6,3,3,3,2,4,4]}
    }
                
    print('Sampling one graph per degree sequence...')
                
    x,_,_ = sample_bipartite_graph(seqs['interbank1']['r'], seqs['interbank1']['c'], rule='most_uniform', dual_method='root')
    print('Interbank1 done')
    
    x,_,_ = sample_bipartite_graph(seqs['interbank2']['r'], seqs['interbank2']['c'], rule='most_uniform', dual_method='root')
    print('Interbank2 done')
    
    x,_,_ = sample_bipartite_graph(seqs['chesapeake']['r'], seqs['chesapeake']['c'], rule='most_uniform', dual_method='root')
    print('Chesapeake done')
    
    x,_,_ = sample_undirected_graph(seqs['chesapeake-und']['d'], rule='most_uniform', dual_method='root')
    print('Chesapeake undirected done')
    
    print('Done!')
    
    # run the following with IPython %timeit
#    %timeit sample_bipartite_graph(seqs['interbank1']['r'], seqs['interbank1']['c'], rule='most_uniform', dual_method='root')
#    167 ms ± 8.48 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#
#    %timeit sample_bipartite_graph(seqs['interbank2']['r'], seqs['interbank2']['c'], rule='most_uniform', dual_method='root')
#    146 ms ± 10.3 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#
#    %timeit sample_bipartite_graph(seqs['chesapeake']['r'], seqs['chesapeake']['c'], rule='most_uniform', dual_method='root')
#    2.24 s ± 57.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
#
#    %timeit sample_undirected_graph(seqs['chesapeake-und']['d'], rule='most_uniform', dual_method='root')
#    1.05 s ± 42.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    