#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:21:56 2021

@author: Enrique
"""

''' Implementation of Erdos-Renyi '''

import numpy as np

def sample_erdos_renyi(prob, num_nodes, graph_type, num_samples):
    ''' 
    
    Sample graphs using the Erdos-Renyi model.
    
    Parameters
    ----------
    prob: probability of edge occurring
    num_nodes: number of nodes in graph, tuple for bipartite graphs
    graph_type: one of 'undirected', 'bipartite'
    num_samples: number of sampled graphs
    
    Returns
    -------
    graphs: array with encoded graphs, size num_samples
    
    '''
    
    assert graph_type in ['undirected', 'bipartite']
    
    if graph_type == 'undirected':
        num_edges = num_nodes*(num_nodes - 1)//2
    else:
        num_edges = num_nodes[0]*num_nodes[1]
    graphs = (np.random.rand(num_samples, num_edges) < prob).astype(int)
    if num_samples == 1:
        graphs = graphs[0]
    return graphs

def best_er_probability(degree_seq):
    '''
    
    Compute the best Erdos-Renyi probability for sampling from the 
    target set (undirected case).
    
    Parameters
    ----------
    degree_seq: vector with the desired degree sequence
    
    Returns
    -------
    prob: best Erdos-Renyi probability
    
    '''
    
    num_nodes = len(degree_seq)
    num_edges = num_nodes*(num_nodes - 1)//2
    d_bar = np.sum(degree_seq)
    prob = 0.5*d_bar/num_edges # fraction of edges present
    return prob
    

if __name__ == '__main__':
    
    g1 = sample_erdos_renyi(prob=0.5, num_nodes=5, graph_type='undirected',
                            num_samples=10)
    print(g1)
    
    g2 = sample_erdos_renyi(prob=0.5, num_nodes=[2, 3], graph_type='bipartite',
                            num_samples=10)
    print(g2)
    