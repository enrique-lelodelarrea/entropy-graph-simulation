#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 18:29:57 2021

@author: Enrique
"""

import logging

DATE_FMT = '%b-%d-%y %H:%M:%S'
LOG_FOLDER = 'logs'

def get_console_handler(level=logging.INFO):
    ch = logging.StreamHandler()
    ch.setLevel(level)
    c_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(c_formatter)
    return ch

def get_file_handler(log_file, level=logging.INFO, file_mode='w'):
    fh = logging.FileHandler(log_file, mode=file_mode)
    fh.setLevel(level)
    f_formatter = logging.Formatter('%(name)s - %(asctime)s - %(levelname)s - %(message)s',
                                    datefmt=DATE_FMT)
    fh.setFormatter(f_formatter)
    return fh

def create_logger_w_c_handler(logger_name, logger_level=logging.INFO):
    ''' Create a logger with a console handler. '''
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level) # this determines which commands are logged
    logger.addHandler(get_console_handler(logger_level)) # level of handler and logger match
    return logger

def create_logger_w_handlers(logger_name, log_file_name, logger_level=logging.INFO):
    ''' Create a logger object with two handlers, one for logging in the 
    terminal and another one for logging into a file.'''
    # create a custom logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level) # this determines which commands are logged
    # add handlers to the logger
    logger.addHandler(get_console_handler(logger_level))
    logger.addHandler(get_file_handler(log_file_name, logger_level))
    return(logger)
    
def close_handlers(logger):
    ''' Close the logger and all its handlers. '''
    handlers = logger.handlers[:] # creates a copy of the list, ow you skip elements
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)