#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:06:51 2021

@author: Enrique
"""

''' Functions to solve the maximum entropy problem in the binary case. '''

import logging
import numpy as np
# solvers for max entropy problem
import cvxpy as cvx
from scipy.optimize import root # for the dual system of eqs

# aux functions
from .aux_functions import lambda_to_eta

# module logger
logger = logging.getLogger(__name__)

def exp_value_exponential_safe(eta, critical_val = 100):
    ''' 
    
    Compute the expected value of X under the exponential distribution.
    In the binary case, we have a Bernoulli with p = exp(eta).
    
    Avoids numpy overflow.
    
    Parameters
    ----------
    eta: numeric vector of natural parameters
    critical_val: critical value to avoid overflow

    Returns
    -------
    mean: numeric vector of means
    
    '''
    
    mask = np.logical_or(eta > critical_val, eta < -critical_val)
    mean = np.empty_like(eta, dtype=float)
    mean[eta < -critical_val] = 0.
    mean[eta > critical_val] = 1.
    e_eta = np.exp(eta[~mask]) # etas are in a reasonable range
    temp = e_eta/(1. + e_eta) # no overflow should happen
    mean[~mask] = temp
    return mean
    

# TODO: round up to 0 or 1
def exp_value_exponential(eta):
    ''' 
    
    Compute the expected value of X under the exponential distribution.
    In the binary case, we have a Bernoulli with p = exp(eta).
    
    Parameters
    ----------
    eta: numeric vector of natural parameters

    Returns
    -------
    mean: numeric vector of means
    
    '''
    
    e_eta = np.exp(eta) # etas are assumed to be in a reasonable range
    mean = e_eta/(1. + e_eta)
    return mean


def eval_dual_eqs(lbda, A, b):
    ''' 
    
    Compute the error of the dual non linear system.
    
    Parameters
    ----------
    lbda: numeric vector, dual variables
    A: numeric matrix
    b: integer vector
    
    Returns
    -------
    error: numeric vector of errors
    
    '''
    eta = lambda_to_eta(lbda, A)
    means = exp_value_exponential_safe(eta) # avoid overflow (slower)
#    means = exp_value_exponential(eta) # can overflow
    error = np.dot(A, means) - b
    return error

def discrete_max_entr_dual(A, b, method='cvxpy'):
    ''' 
    
    Solve the dual of the maximum entropy problem for the 
    binary case.
    
    Parameters
    ----------
    A: constraint matrix (m x n)
    b: constraint rhs (m-dimensional) vector 
    method: string, either 'cvxpy' or 'root'
    
    Returns
    -------
    lbda: an m-dimensional vector with the optimal dual variables
    
    '''
    
    assert method in ['cvxpy', 'root']
    
    # number of constraints and variables
    if len(A.shape) == 1:
        A = A[np.newaxis, :]
    [m, n] = A.shape
    # solve the dual using a convex opt solver
    if method == 'cvxpy':
        # set up an optimization problem
        # decision variables
        x = cvx.Variable(shape=m) # one variable per row
        x.value = np.zeros(m) # start them at zero
        # objective function (dual problem)
        obj = cvx.sum(cvx.logistic(A.T*x)) - b*x 
        # solve problem
        prob = cvx.Problem(cvx.Minimize(obj))
        prob.solve()
        if prob.status != 'optimal':
            logger.warning('CVXPY did not converge!')
            logger.warning('Problem status is: %s' % prob.status)
        if x.value is None: # no solution
            lbda = None
        else:
            # recover solution in numpy array
            lbda = np.array(x.value)
    # solve the dual by solving the FOC directly
    elif method == 'root':
        # solve the dual non linear system directly (faster than cvxpy but maybe less robust)
        res = root(eval_dual_eqs, np.zeros(m), args=(A, b))
        if not res.success:
            # even if it does not converge, lbda can be used
            logger.info('Scipy root did not converge!')
            logger.info('Problem status is: %s' % res.message)
        lbda = res.x
    return lbda
    
    

