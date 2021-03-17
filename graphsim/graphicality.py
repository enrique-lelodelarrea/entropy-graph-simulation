#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:16:51 2021

@author: Enrique
"""

'''
Several tests for the graphicality of a degree sequence.

These are naive implementations. One can use instead the
functions of the NetworkX library.
'''

import numpy as np

def sequence_is_even(deg_seq):
    ''' Checks if the elements of the degree sequence deg_seq
    add up to an even number. '''
    sum_deg = np.sum(deg_seq)
    return (sum_deg%2 == 0)

def order_seq_descending(deg_seq):
    ''' Orders a degree sequence in descending order.
    Ordering is done in place. '''
    deg_seq[::-1].sort() # interesting behavior of numpy views
    return None

def erdos_gallai(deg_seq):
    ''' Naive implementation of Erdos-Gallai test.
    Test for the graphicality of a simple undirected graph.'''
    # check if even
    if not sequence_is_even(deg_seq):
        return False # degrees have to sum up to an even number
    # sort deg_seq
    order_seq_descending(deg_seq)
    # iterate (maybe can be done in vector form)
    cum_sum_d = np.cumsum(deg_seq)
    n = len(deg_seq)
    k_vec = np.arange(n)
    k_times_k_plus_1 = k_vec*(k_vec + 1)
    for k in range(n):
        sum_min_k_d = np.sum(np.minimum(k + 1, deg_seq[k+1:]))
        if cum_sum_d[k] > k_times_k_plus_1[k] + sum_min_k_d:
            return False # condition is failed for one k
    return True

def erdos_gallai_vec(deg_seq):
    ''' Naive implementation of Erdos-Gallai test.
    Test for the graphicality of a simple undirected graph.
    Avoids Python loops.'''
    # check if even
    if not sequence_is_even(deg_seq):
        return False # degrees have to sum up to an even number
    # sort deg_seq
    order_seq_descending(deg_seq)
    # iterate (maybe can be done in vector form)
    cum_sum_d = np.cumsum(deg_seq)
    n = len(deg_seq)
    k_vec = np.arange(n) + 1 # starting from 1
    k_times_k_minus_1 = k_vec*(k_vec - 1)
    min_k_d = np.triu(np.minimum(deg_seq, k_vec[:,np.newaxis]), k=1)
    sum_min_k_d = np.sum(min_k_d, axis=1)
    # erdos gallai condition, vector form
    erdos_gallai_cond = (cum_sum_d <= k_times_k_minus_1 + sum_min_k_d)
    if np.all(erdos_gallai_cond):
        return True
    else:
        return False
    

if __name__ == '__main__':
    
    seq1 = np.array([4, 5, 6, 7, 2], dtype=int)
    seq2 = np.array([], dtype=int) # Empty graph
    seq3 = np.array([3, 2, 2, 2, 1], dtype=int) # B and D example 
    seq4 = np.array([7,8,5,1,1,2,8,10,4,2,4,5,3,6,7,3,2,7,6,
                     1,2,9,6,1,3,4,6,3,3,3,2,4,4])
    
    # test Erdos-Gallai
    print(erdos_gallai(seq1))
    print(erdos_gallai(seq2))
    print(erdos_gallai(seq3))
    print(erdos_gallai(seq4))
    print('')
    print(erdos_gallai_vec(seq1))
    print(erdos_gallai_vec(seq2))
    print(erdos_gallai_vec(seq3))
    print(erdos_gallai_vec(seq4))
    
    
    