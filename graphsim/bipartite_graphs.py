#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 17:32:18 2021

@author: Enrique
"""

''' Functions to sample bipartite graphs with prescribed
degree sequences, using the sequential algorithm. '''

import numpy as np
from .sequential_algo_binary import seq_algo_binary

def bipartite_graph_matrix(m, n):
    ''' 
    
    Construct the linear constraint matrix for the 
    bipartite graph problem.
    
    Parameters
    ----------
    m: number of rows (nodes of first set)
    n: number of columns (nodes of second set)
    
    Returns
    -------
    A: matrix of size (m+n)x(mn) 
    
    '''
    A = np.zeros([m + n, m*n], dtype=int)
    A[:m,:] = np.kron(np.eye(m), np.ones(n))
    A[m:,:] = np.kron(np.ones(m), np.eye(n))
    return A

def bipartite_graph_to_binary_instance(r, c):
    ''' 
    
    Compute the affine mapping of the bipartite graph instance.
    
    Parameters
    ----------
    r: row degrees (nodes of first set)
    c: column degrees (nodes of second set)
    
    Returns
    -------
    A: matrix of linear mapping
    b: rhs vector of linear mapping
    
    '''
    m = len(r)
    n = len(c)
    A = bipartite_graph_matrix(m, n)
    b = np.concatenate((r, c))
    return A, b

def sample_bipartite_graph(r, c, rule='fixed', dual_method='cvxpy'):
    ''' 
    
    Simulate a random bipartite graph x which satisfies the given
    degree sequences using the sequential maximum entropy algorithm.
    
    Parameters
    ----------
    r: row degrees (nodes of first set), length m
    c: column degrees (nodes of second set), length n
    rule: string, edge selection rule
    dual_method: string, method for solving the dual of the maximum entropy problem
    
    Returns
    -------
    x: edge indicator vector of size mxn (1 if edge is present)
    p: numeric, estimator of p_x, where p_x is the probability of observing x
    w: numeric, estimator of 1/p_x
    
    '''
    
    # check feasibility
    # TODO: implement Gale Ryser or some other criterion
    # I don't see it in NetworkX
    
    # create linear mapping
    (A, b) = bipartite_graph_to_binary_instance(r, c)
    
    # call binary sequential algo
    x, p, w = seq_algo_binary(A, b, rule=rule, dual_method=dual_method)
    return x, p, w

def vector_to_matrix(x, m, n):
    ''' 
    
    Transform a vector in long form into the original mxn matrix.
    
    Parameters
    ----------
    x: mxn vector
    m: number of rows
    n: number of columns
    
    Returns
    -------
    X: mxn matrix with the elements of x 
    
    '''
    if len(x) != m*n:
        raise ValueError('length of x is not equal to m*n')
    X = np.reshape(x, (m,n))
    return X
