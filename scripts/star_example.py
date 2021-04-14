#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:55:40 2021

@author: Enrique
"""

''' Probability of acceptance for optimal ER for a star example.

Consider a graph with M nodes and n=(M choose 2) edges.
A star degree sequence is of the form d = ((M-1), 1, ..., 1).
The degree sum is: 2*(M-1).
Optimal ER probability is then beta = sum(d)/(2n) = (M-1)/(M choose 2) = 2/M.

For a general beta, the probability of sampling a graph is:
    
    P(graph) = prod_{i < j} beta**x_ij * (1 - beta)**(1 - x_ij).
    
If the graph satisfiies the degree sequence d, we have constraint on the sum
of the x_ij:
    
    P(graph) = beta**(sum(d)/2) * (1 - beta)**(n - sum(d)/2)

'''

def prob_star(M):
    '''
    
    Compute the probability of observing the star with ER and best
    probability.
    
    Parameters
    ----------
    M: number of nodes
    
    Returns
    -------
    prob: probability of observing the star
    
    '''
    if isinstance(M, int):
        assert M > 2
    else:
        assert np.all(M > 2)
    beta = 2/M
    n = M*(M - 1)//2
    prob = (1 - beta)**n * (beta/(1 - beta))**(M - 1)
    return prob

if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    for m in [3, 4, 5, 6, 7, 8]:
        print('Prob = %s' % prob_star(m))
    
    m_vec = np.arange(10, dtype=int) + 3
    probs = prob_star(m_vec)
    
#    plt.plot(m_vec, probs)
    
    print()
    for m in [4, 5]:
        print('Size of star is %s' % m)
        p = prob_star(m)
        print('Prob = %.4f' % p)
        print('ME/RE-1 = %.4f' % (1./p - 1))
    

    
    