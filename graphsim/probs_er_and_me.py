#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:06:59 2021

@author: Enrique
"""

import numpy as np
import logging
from scipy.special import entr
from .erdos_renyi import best_er_probability
from .aux_functions import lambda_to_eta
from .max_ent_binary_case import discrete_max_entr_dual, exp_value_exponential_safe
from .undirected_graphs import graph_matrix as und_graph_matrix

# module logger
logger = logging.getLogger(__name__)

def graph_prob_ER(p, deg_seq):
    '''
    
    Compute the probability of observing a graph with ER.
    The graph satisfies the deg_seq provided.
    
    Parameters
    ----------
    p: edge probability of ER model
    deg_seq: degree sequence (assumed to be graphical)
    
    Returns
    -------
    prob: probability of observing a graph with fixed degree sequence
    
    '''
    d_bar = sum(deg_seq)
    num_nodes = len(deg_seq)
    assert d_bar%2 == 0 # check that it is even
    num_edges = num_nodes*(num_nodes - 1)//2
    prob = p**(d_bar//2) * (1 - p)**(num_edges - d_bar//2)
    logger.info('ER prob is %g' % prob)
    return prob

def graph_prob_ME(p):
    '''
    
    Compute the probability of observing a graph with ME.
    The graph satisfies the deg_seq provided. The prob is exp(-H).
    
    Parameters
    ----------
    p: vector with max entropy probabilities
    
    Returns
    -------
    prob: probability of observing a graph with fixed degree sequence
    
    '''
    # compute entropy
    entropy = np.sum(entr(p) + entr(1. - p))
    prob = np.exp(-entropy)
    logger.info('ME prob is %g' % prob)
    return prob

def ratio_ME_to_ER(deg_seq):
    '''
    
    Compute the ratio of acceptance probabilities, ME vs ER.
    
    Parameters
    ----------
    deg_seq: degree sequence (assumed to be graphical)
    
    Returns
    -------
    ratio: ratio of probabilities
    
    '''
    # prob Erdos Renyi
    p_ER = best_er_probability(deg_seq) # edge probability
    prob_ER = graph_prob_ER(p_ER, deg_seq)
    
    # prob Max Entropy
    
    A = und_graph_matrix(len(deg_seq))    
    # solve max entropy problem
    lbda = discrete_max_entr_dual(A, deg_seq, method='cvxpy')
    eta = lambda_to_eta(lbda, A)
    logger.debug('eta = %s' % eta)
    p_vec_ME = exp_value_exponential_safe(eta, critical_val=20)
    
    prob_ME = graph_prob_ME(p_vec_ME)
    
    # ratio
    ratio = prob_ME / prob_ER
    
    return ratio

def acceptance_prob_ME(deg_seq, cardinality):
    '''
    
    Compute the acceptance probability of AR with ME.
    
    Parameters
    ----------
    deg_seq: degree sequence (assumed to be graphical)
    cardinality: size of the target set
    
    Returns
    -------
    prob: acceptance probability
    
    '''
    # prob Max Entropy
    A = und_graph_matrix(len(deg_seq))    
    # solve max entropy problem
    lbda = discrete_max_entr_dual(A, deg_seq, method='cvxpy')
    eta = lambda_to_eta(lbda, A)
    logger.debug('eta = %s' % eta)
    p_vec_ME = exp_value_exponential_safe(eta, critical_val=20)
    prob_ME = graph_prob_ME(p_vec_ME)
    
    prob = cardinality*prob_ME
    
    return prob
    
    