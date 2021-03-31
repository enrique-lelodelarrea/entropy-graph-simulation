#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:33:15 2021

@author: Enrique
"""

# %% Libraries and parameters

from read_sims import read_sims_file, weights_scaling_stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

# use latex fonts
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

graph_name = 'chesapeake'
prefix = 'ches'

# folder and file parameter
sub_folder = '../sims/' + graph_name + '/'
p_file = prefix

# histogram parameters
hist_name = '../figs/' + p_file + '_hist_schemes.pdf'
hist_name_pgf = '../figs/' + p_file + '_hist_schemes.pgf'
f_size = 9 # font size for title and axis
t_size = 8 # font size for ticks
grid = False

# table parameters
table_name = '../figs/' + p_file + '_stats.html'

# %% Reading the data

rule_to_idx = {'fixed': [1, 2, 3, 4, 5, 6],
               'random': [7, 8, 9, 10, 11, 12],
               'most_uniform': [13, 14, 15, 16, 17, 18],
               'most_singular': [19, 20, 21, 22, 23, 24]}

file_sizes = 1000
scale_factor = 1e57
filter_succ = True # remove failed samples

num_sims = 5000
#1e57 for chesapeake
#num_sims = None # set to None if you want to use all the simulations


################## fixed
rule = 'fixed'
# read the data
X0, w0 = read_sims_file(sub_folder, p_file, rule, file_sizes,
                        rule_to_idx[rule], filter_succ, num_sims)
# scale weights
w_scaled_0, stats_0 = weights_scaling_stats(w0, scale_factor)
stats_0['rule'] = rule
################## random
rule = 'random'
# read the data
X1, w1 = read_sims_file(sub_folder, p_file, rule, file_sizes,
                        rule_to_idx[rule], filter_succ, num_sims)
# scale weights
w_scaled_1, stats_1 = weights_scaling_stats(w1, scale_factor)
stats_1['rule'] = rule
################## most_uniform
rule = 'most_uniform'
# read the data
X2, w2 = read_sims_file(sub_folder, p_file, rule, file_sizes,
                        rule_to_idx[rule], filter_succ, num_sims)
# scale weights
w_scaled_2, stats_2 = weights_scaling_stats(w2, scale_factor)
stats_2['rule'] = rule
################## most_singular
rule = 'most_singular'
# read the data
X3, w3 = read_sims_file(sub_folder, p_file, rule, file_sizes,
                        rule_to_idx[rule], filter_succ, num_sims)
# scale weights
w_scaled_3, stats_3 = weights_scaling_stats(w3, scale_factor)
stats_3['rule'] = rule

# %% Plot histograms

dict_graph_name = {
        'chesapeake':'CH',
        'interbank':'IB-1',
        'interbank2':'IB-2'}

dict_scheme_name = {
        'fixed':'F',
        'random':'RN',
        'most_uniform':'MU',
        'most_singular':'MS'}

def plot_histogram(graph_name, weights, num_bins, density, alpha, color, scale_factor, fontsize, labelsize, rule, grid):
    ''' Plots a histogram.
    Parameters
    ----------
    fontsize: integer, size of title of graph and axis
    labelsize: integer, size of ticks
    '''
    
    plt.hist(weights, num_bins, density=density, alpha=alpha, color=color)
    plt.xlabel('weight/%g'%scale_factor, fontsize=fontsize)
    if density:
        y_label = 'Relative Frequency'
    else:
        y_label = 'Frequency'
    plt.ylabel(y_label, fontsize=fontsize)
    #plt.title('Graph: %s, Scheme: %s, Sample Size: %i'%(dict_graph_name[graph_name],
    #                                                    dict_scheme_name[rule], len(weights)), fontsize=fontsize)
    #plt.title('%s, $m=%i$'%(dict_scheme_name[rule], len(weights)), fontsize=fontsize)
    plt.title('%s' % (dict_scheme_name[rule]), fontsize=fontsize)
    
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([0, 55, 0, 350])
    if grid:
        plt.grid(True)
    plt.tick_params(labelsize=labelsize)
    plt.tight_layout()
    return None
    
# print in grayscale
plt.style.use('grayscale')
color = 'C1'

# plot histogram
scale = 0.95
plt.figure(figsize=(scale*6.0,scale*4.0))
#plt.figure()

plt.subplot(221)
rule = 'fixed'
plot_histogram(p_file, w_scaled_0, 50, False, 0.75, color, scale_factor, f_size, t_size, rule, grid)

plt.subplot(222)
rule = 'random'
plot_histogram(p_file, w_scaled_1, 50, False, 0.75, color, scale_factor, f_size, t_size, rule, grid)

plt.subplot(223)
rule = 'most_uniform'
plot_histogram(p_file, w_scaled_2, 50, False, 0.75, color, scale_factor, f_size, t_size, rule, grid)

plt.subplot(224)
rule = 'most_singular'
plot_histogram(p_file, w_scaled_3, 50, False, 0.75, color, scale_factor, f_size, t_size, rule, grid)

print('saving ' + str(hist_name))
plt.savefig(hist_name)
plt.savefig(hist_name_pgf)

# %% Summary statistics

# data frame with summary stats

df_stats = pd.DataFrame(columns=['rule', 'mean', 'min', 'max', 'median', 'sd',
                                 'max/median', 'max/sum'])
df_stats = df_stats.append(stats_0, ignore_index=True)
df_stats = df_stats.append(stats_1, ignore_index=True)
df_stats = df_stats.append(stats_2, ignore_index=True)
df_stats = df_stats.append(stats_3, ignore_index=True)

df_stats.to_html(table_name)

print(df_stats)

#print(df_stats.to_latex())

def print_data_latex(array, sci_not = False):
    ''' Transforms a numeric vector into one string, separated by "&"s '''
    string_latex = ''
    for i in range(len(array)):
        if i > 0:
            string_latex += ' & '
        if sci_not:
            num_to_str = "{:.1e}".format(array[i])
        else:
            num_to_str = str(np.round(array[i], 1))
        string_latex += num_to_str
    return string_latex

d1 = np.array(df_stats['max/median'])
s1 = print_data_latex(d1, False)

d2 = np.array(df_stats['max/sum'])
s2 = print_data_latex(d2, True)


print(s1)
print(s2)