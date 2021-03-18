#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:19:19 2021

@author: Enrique
"""

''' Auxiliary functions. '''

import numpy as np

def lambda_to_eta(lbda, A):
    ''' 
    
    Compute the parameter of the exponential families eta = lbda^T * A.
    
    Parameters
    ----------
    lbda: m vector of dual variables
    A: mxn matrix of linear constraints
    
    Returns
    -------
    eta: n vector of exponential parameters 
    
    '''
    eta = np.dot(A.T, lbda)
    return eta

def sat_linear_constraints(x, A, b):
    ''' 
    
    Check if the point x satisfies the constraints Ax = b.
    
    Parameters
    ----------
    x: n vector
    A: mxn matrix
    b: m vector
    
    Returns
    -------
    sat: boolean, True if Ax = b, False o.w. 
    
    '''
    abs_diff = np.abs(np.dot(A, x) - b)
    if np.all(abs_diff < 1e-3):
        sat = True
    else:
        sat = False
    return sat

def next_coordinate(list_coordinates, rule='fixed', eta_vec=None):
    ''' 
    
    Selects the next edge for the sequential algorithm.
    
    Parameters
    ----------
    list_coordinates: n vector with remaining coordinates (indices)
    rule: one of "fixed", "random", "most_uniform", "most_singular",
        determines how to pick the next entry
    eta_vec: vector of parameters for the exponential families 
    
    Returns
    -------
    next_i: integer, the index of the next entry in list_coordinates
    orig_i: integer, the index of the next entry wrt all coordinates
    
    '''
    if rule == 'fixed':
        next_i = 0
    elif rule == 'random':
        n = len(list_coordinates)
        next_i = np.floor(n*np.random.rand()).astype(int) # random integer
    elif rule == 'most_uniform':
        # select the entry with the eta value closest to zero
        next_i = np.argmin(np.abs(eta_vec))
    elif rule == 'most_singular':
        # select the entry with the largest eta value in absolute value
        next_i = np.argmax(np.abs(eta_vec))
    else:
        raise ValueError("Value of rule is not one of: 'fixed', 'random', 'most_uniform', or 'most_singular'!")
    orig_i = list_coordinates[next_i] 
    return next_i, orig_i