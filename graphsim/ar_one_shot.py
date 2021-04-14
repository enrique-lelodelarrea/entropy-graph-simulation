#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:34:58 2021

@author: Enrique
"""

''' Acceptance rejection algorithms. '''

MAX_TRIES = 1000

import logging
import sys
import numpy as np
import networkx as nx
from .erdos_renyi import sample_erdos_renyi, best_er_probability
from .undirected_graphs import graph_matrix as und_graph_matrix
from .aux_functions import sat_linear_constraints, lambda_to_eta
from .max_ent_binary_case import discrete_max_entr_dual, exp_value_exponential_safe

# module logger
logger = logging.getLogger(__name__)

def ar_one_shot(degree_seq, graph_type, method, prob=None, max_num_tries=None,
                num_samples=1, print_progress=False):
    '''
    
    Sample graphs with prescribed degrees using acceptance rejection
    via either Erdos-Renyi (naive) or maximum entropy.
    
    Parameters
    ----------
    degree_seq: desired degree sequence
    num_nodes: number of nodes in graph, tuple for bipartite graphs
    graph_type: one of 'undirected', 'bipartite'
    method: one of 'erdos-renyi', 'max-entropy'
    prob: probability of edge occurring, only for erdos-renyi, if None
        it computes the optimal prob for sampling from target set
    max_num_tries: number of tries allowed
    num_samples: number of graphs desired
    print_progress: print progress every 100 iterations
    
    Returns
    -------
    graphs: array with encoded graphs
    num_tries_vec: vector number of tries needed per sample
    
    '''
    assert method in ['erdos-renyi', 'max-entropy']
    assert graph_type in ['undirected', 'bipartite']
    if max_num_tries is None:
        max_num_tries = MAX_TRIES
    # compute constraint matrix
    if graph_type == 'undirected':
        # check graphicality
        if not nx.is_graphical(degree_seq):
            logger.critical('Degree sequence is not graphical!')
            sys.exit()
        num_nodes = len(degree_seq)
        A = und_graph_matrix(num_nodes)
        b = degree_seq
    else:
        num_nodes = (len(degree_seq[0]), len(degree_seq[1]))
        # TODO: complete for bipartite case
    if method == 'max-entropy':
        # solve max entropy problem
        lbda = discrete_max_entr_dual(A, b, method='cvxpy')
        eta = lambda_to_eta(lbda, A)
        logger.debug('eta = %s' % eta)
        probs = exp_value_exponential_safe(eta, critical_val=20)
        logger.info('Probs are: %s' % probs)
    elif method == 'erdos-renyi':
        if prob is None:
            prob = best_er_probability(degree_seq)
            logger.info('Using best ER probability: %s' % prob)
        else:
            logger.info('Using user ER probability: %s' % prob)
    # start sampling
    num_tries_vec = list()
    graphs = list()
    for i in range(num_samples):            
        num_tries = 0
        while True:
            num_tries += 1
            if num_tries > max_num_tries:
                logger.error('Max number of tries (%d) reached!' % max_num_tries)
                graph = None
                sys.exit()
                break
            if method == 'erdos-renyi':
                candidate = sample_erdos_renyi(prob, num_nodes, graph_type, 1)
            elif method == 'max-entropy':
                num_edges = len(probs)
                # simulate edges
                candidate = (np.random.rand(num_edges) < probs).astype(int)
            if sat_linear_constraints(candidate, A, b):
                graph = candidate
                break
        graphs.append(graph)
        num_tries_vec.append(num_tries)
        if (i+1)%100 == 0:
            if print_progress:
                print('i = %s' % (i+1))
    if num_samples == 1:
        return (graph, num_tries)
    return (graphs, num_tries_vec)

if __name__ == '__main__':
    
    pass
        
        
            
    