#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:40:14 2021

@author: Enrique
"""

''' Functions to sample undirected graphs with prescribed
degree sequences, using the sequential algorithm. '''

import sys
import logging
import numpy as np
import networkx as nx
from itertools import combinations

from .max_ent_binary_case import discrete_max_entr_dual, exp_value_exponential
from .aux_functions import lambda_to_eta, sat_linear_constraints, next_coordinate

CRITICAL_ETA_TOL = 1e-6

# module logger
logger = logging.getLogger(__name__)

def graph_matrix(num_nodes):
    ''' 
    
    Construct the linear constraint matrix for the 
    undirected graph problem.
    
    Parameters
    ----------
    num_nodes: number of graph nodes/vertices 
    
    Returns
    -------
    A: matrix of size mxn, where m=num_nodes and n=choose(m,2) 
    
    '''
    assert num_nodes >= 2
    # create fully connected graph instance
    g = nx.complete_graph(num_nodes)
    # get the incidence matrix
    A_sparse = nx.incidence_matrix(g)
    A = A_sparse.toarray().astype(int) # returns a dense array
    return A

def critical_eta(epsilon):
    ''' 
    
    Compute the critical eta that will force w.p.1 an entry 
    to either 0 (eta < 0) or 1 (eta > 0).
    
    Parameters
    ----------
    epsilon: real in 0-1, tolerance level for the probability
    
    Returns
    -------
    eta_crit: the critical eta value (equal to -logit(epsilon))
    
    '''
    eta_crit = -np.log(epsilon/(1 - epsilon))
    return eta_crit

def update_A_b(A, b, idx, value_vec):
    ''' 
    
    Update the matrix A and vector b after fixing certain values.
    
    Parameters
    ----------
    A: linear constraint matrix
    b: rhs vector
    idx: vector of indices indicating which entries have been fixed
    value_vec: vector of values, same size as idx
    
    Returns
    -------
    updated version of A and b 
    
    '''
    b = b - np.dot(A[:,idx], value_vec)
    A = np.delete(A, idx, axis=1)
    return A, b

def remove_constraints(A, b):
    ''' 
    
    Remove redundant constraints from a given sub-instance.
    
    Parameters
    ----------
    A: linear constraint matrix
    b: rhs vector
    
    Returns
    -------
    updated version of A and b
    trivial_status: boolean, can sample the rest of entries uniformly
    
    '''
    
    # FIXME: A is assumed to have 0--1 elements when checking rows with all zeros
    # OK in graph case

    (m,n) = A.shape
    trivial_status = False # turn into True if A and b are zero (trivial constraints)
    # remove the constraints that have all coefficients equal to zero
    A_row_sums = np.sum(A, axis=1)
    A_zero_mask = (A_row_sums == 0)
    n_constraints = np.sum(A_zero_mask)
    logger.info('Removing %d constraints...' % n_constraints)
    if np.any(b[A_zero_mask]):
        logger.error('Infeasible constraint found!')
        sys.exit()
    if n_constraints == m:
        trivial_status = True
    # update constraints
    A = A[~A_zero_mask,:]
    b = b[~A_zero_mask]
    return A, b, trivial_status

def sample_edges(n, eta):
    ''' 
    
    Simulate n samples of the discrete exponential family (Bernoulli)
    via inverse transform.
    
    Parameters
    ----------
    n: integer, number of samples
    eta: real, parameter of the exponential family, p_x \propto exp(eta*x)
    
    Returns
    -------
    x: np array of size n, the random sample 
    
    '''
    u = np.random.rand(n)
    prob = exp_value_exponential(eta) # prob of observing the edge
    x = (u < prob)
    if n == 1:
        x = x[0]
    return x.astype(int)

def p_exponential(x, eta):
    ''' 
    
    Probability that X = x under the exponential marginal
    
    Parameters
    ----------
    x: either 0 or 1
    eta: real
    
    Returns
    -------
    p: probability
    
    '''
    assert x in [0, 1]
    if x == 0:
        p = 1. - exp_value_exponential(eta)
    else: # x is 1
        p = exp_value_exponential(eta)
    return p

def preprocess_instance(A, b, x, remain_coord, pi_x):
    ''' 
    
    Check if certain constraints are redundant and remove them.
    Check if the simulation can be ended early.
    
    Parameters
    ----------
    A: linear constraint matrix
    b: rhs vector
    x: decision variable vector
    remain_coord: entries of x that are yet to be sampled
    pi_x: current lower bound of the probability of observing x
    
    Returns
    -------
    updated versions of A, b, x, pi_x
    early_exit: boolean, True if simulation can be ended
    
    '''
    
    logger.info('Preprocessing current sub-instance...')
    early_exit = False
    # remove redundant rows of A and b
    A, b, trivial_status = remove_constraints(A, b)
    logger.debug('A = ')
    logger.debug(A)
    logger.debug('b = ')
    logger.debug(b)
    if trivial_status:
        logger.info('Remaining constraints are trivially satisfied (using uniform)!')
        # the constraints are always satisfied
        # the solution of the max entropy problem
        # is the uniform distribution
        x[remain_coord] = sample_edges(n=len(remain_coord), eta=0.)
        pi_x *= 0.5**(len(remain_coord)) # tau and chi are not necessary
        early_exit = True
    # the last iteration is trivial
    elif len(remain_coord) == 1:
        logger.info('Only one coordinate remaining...')
        x[remain_coord[0]] = b[0]/A[0,0]
        pi_x *= 1 # there is no estimation in this case (tau and chi are not necessary)
        early_exit = True
    return A, b, x, pi_x, early_exit

def sample_undirected_graph(deg_seq, rule='fixed', dual_method='cvxpy'):
    ''' 
    
    Simulate a random graph x which satisfies the given
    degree sequences using the sequential maximum entropy algorithm.
    
    Parameters
    ----------
    deg_sequence: prescribed degree sequence; n vector
    rule: string, edge selection rule
    dual_method: string, method for solving the dual of the maximum entropy problem
    
    Returns
    -------
    x: edge indicator vector of size n*(n-1)/2 (1 if edge is present)
    pi_x: real, lower bound for the probability (p_x) of drawing x using the algorithm,
    equal to the actual probability in the binary case
    tau_x: real, factor such that tau_x*pi_x is an unbiased estimator of p_x
    chi_x: real, factor such that chi_x/pi_x is an unbaised estimator of 1/p_x
    p: numeric, estimator of p_x, where p_x is the probability of observing x
    w: numeric, estimator of 1/p_x
    
    '''
    
    # some parameters
    # threshold of eta that yield prob 0 or 1
    eta_critical = critical_eta(CRITICAL_ETA_TOL)
    num_nodes = len(deg_seq)
    assert num_nodes > 1 # at least two nodes
    # order the degree sequence and check if it's graphical
    if not nx.is_graphical(deg_seq):
        logger.error('Initial degree sequence is not graphical!')
        sys.exit()
    # compute graph incidence matrix
    inc_matrix = graph_matrix(num_nodes)
    # number of constraints and variables
    assert len(inc_matrix.shape) == 2
    [m, n] = inc_matrix.shape
    # allocate output vector
    x = np.zeros(n, dtype=int)
    # initial probability estimators
    pi_x = 1.
    tau_x = 1.
    chi_x = 1.
    # remaining edges to be simulated
    remain_coord = np.arange(n, dtype=int)
    # copy of parameters
    A = np.copy(inc_matrix)
    b = np.copy(np.array(deg_seq))
    logger.info('Start of sequential algorithm')
    logger.info('Critical eta is: %g' % eta_critical)
    logger.info('Selection rule is: %s' % rule)
    # enter the loop
    while(len(remain_coord) > 0):
        logger.info('Number of remaining coordinates: %s' % len(remain_coord))
        logger.info('Starting iteration...')
        
        # preprocessing, removing redundant constraints
        #A_copy, b_copy, x, remain_coord = remove_trivial_constraints_and_update(A_copy, b_copy, verbose, x, remain_coord, d)
        A, b, x, pi_x, early_exit = preprocess_instance(A, b, x, remain_coord, pi_x)
        if early_exit:
            break  
        # solve the max entropy problem
        logger.info('Solving the max entropy problem...')
        lbda = discrete_max_entr_dual(A, b, dual_method)
        # compute exponential family parameters
        eta = lambda_to_eta(lbda, A)
        logger.debug('Eta star =')
        logger.debug(np.round(eta, 3))
        # fix some entries to zero or one, according to a threshold
        mask_zero = (eta <= -eta_critical)
        mask_one = (eta >= eta_critical)
        mask_both = np.logical_or(mask_zero, mask_one)
        if np.any(mask_both):
            # find the indices to be fixed
            idx_eta_fix = np.where(mask_both)[0]
            idx_eta_zero = np.where(mask_zero)[0]
            fixed_vals = np.ones(len(idx_eta_fix), dtype=int) # some values equal to 1
            fixed_vals[np.isin(idx_eta_fix, idx_eta_zero)] = 0 # the rest equal to zero
            # udpate A and b
            A, b = update_A_b(A, b, idx_eta_fix, fixed_vals)
            # remove fixed entries from eta
            eta = np.delete(eta, idx_eta_fix)
            logger.info('Fixing some entries to one or zero...')
            logger.debug('Current indices fixed:')
            logger.debug(idx_eta_fix)
            logger.debug('Coordinates fixed:')
            logger.debug(remain_coord[idx_eta_fix])
            # add values to final vector and reduce remaining coordinates
            x[remain_coord[idx_eta_fix]] = fixed_vals
            remain_coord = np.delete(remain_coord, idx_eta_fix)
            logger.info('Number of remaining coords (after fixing): %s' % len(remain_coord))
        if len(remain_coord) == 0:
            logger.info('All coordinates have been fixed. Exiting...')
            break
        # preprocessing, removing redundant constraints
        A, b, x, pi_x, early_exit = preprocess_instance(A, b, x, remain_coord, pi_x)
        if early_exit:
            break
        # ready to simulate one edge
        # select the next entry (fixed or adaptive)
        next_i, orig_i = next_coordinate(remain_coord, rule=rule, eta_vec=eta)
        logger.info('Next simulated edge will be %s, with current index %s' % (orig_i, next_i))
        logger.info('Edge has parameter eta = %g' % eta[next_i])
        logger.info('Simulating edge...')
        if True: # no feasibility oracle is required
            # draw a sample for the selected coordinate
            x_i = sample_edges(n=1, eta=eta[next_i])
            # tau_x and chi_x are always one
            tau_vec = np.array([1, 1])
        else:
            pass
            # TODO: add the oracle (not as easy as I thought)
#            tau_vec = np.array([0, 0])
#            for j in range(2):
#                feas_oracle = False
#                while not feas_oracle:
#                    tau_vec[j] += 1
#                    # draw a (tentative) sample for the selected coordinate
#                    x_i = sample_edges(1, eta[next_i])
#                    if verbose: print("Try x_i = %d" % x_i)
#                    A_temp, b_temp = update_A_b(A, b, next_i, x_i)
#                    # check feasibility of sub-instance
#                    feas_oracle = feasibility_oracle(A_temp, b_temp, d, verbose)
        logger.info('Simulated value: x_i = %d' % x_i)
        # update constraints
        A, b = update_A_b(A, b, next_i, x_i)
        logger.info('Updating A and b given x_i...')
        logger.debug('A = ')
        logger.debug(A)
        logger.debug('b = ')
        logger.debug(b)
        # store simulated edge
        x[orig_i] = x_i
        pi_x *= p_exponential(x_i, eta[next_i]) # update probability
        tau_x *= (tau_vec[0] + tau_vec[1])/2.
        chi_x *= 1./(tau_vec[0] + tau_vec[1] - 1)
        logger.debug('pi_x, tau_x, chi_x = ')
        logger.debug(pi_x, tau_x, chi_x)
        # remove coordinate
        remain_coord = np.delete(remain_coord, next_i)
        logger.info('Iteration ended.')
        # end of while loop
    # check correctness
    logger.info('All edges have been simulated. Checking for correctness...')
    if not sat_linear_constraints(x, inc_matrix, deg_seq):
        logger.error('Output x does not satisfy the degree constraints!')
        sys.exit()
        # allow to continue
#        raise ValueError('Output x does not satisfy the degree constraints!')
    else:
        logger.info('Output x is correct')
    # probability estimator and IS weight estimator
    p = tau_x*pi_x
    w = chi_x/pi_x
    return x, p, w

def vector2graph(x, num_nodes):
    '''
    
    Transform a vector of edge indicators into a networkx
    graph instance.
    
    Parameters
    ----------
    x: vector of edge indicators
    num_nodes: number of nodes in the graph
    
    Returns
    -------
    graph: graph object
    
    '''
    num_total_edges = num_nodes*(num_nodes - 1)//2
    assert len(x) == num_total_edges
    graph = nx.Graph()
    # add nodes
    nodes = [(i+1) for i in range(num_nodes)]
    graph.add_nodes_from(nodes)
    # add edges
    all_edges = list(combinations(nodes, 2))
    edges = [all_edges[i] for i in range(num_total_edges) if x[i] == 1]
    graph.add_edges_from(edges)
    return graph

if __name__ == '__main__':
    
    pass
    