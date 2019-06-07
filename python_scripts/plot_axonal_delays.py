#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:24:42 2019

@author: eugene

Script plots pdf and cdf of axonal conduction delay distribution
"""
from show_axonal_conduction_dist import read_connections
from analyze_dynamics import calculate_hist_with_fixed_bin_size
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    filename = "/home/eugene/Programming/data/mlong/uniform/grid10/RA_RA_connections.bin"
    _, _, _, delays = read_connections(filename)
    
    all_delays = [d for out_delays in delays for d in out_delays]
    
    delays_rounded = np.round(all_delays, decimals=3)
# =============================================================================
#     uniqueDelays, counts = np.unique(delays_rounded, return_counts = True)
#     cdf_model = np.cumsum((np.ones(len(uniqueDelays))*counts)/np.float(sum(counts)))   
#      
#     
#     f = plt.figure()
#     plt.step(np.concatenate(([0],uniqueDelays)), np.concatenate(([0],cdf_model)), where='post', label='result')
#     plt.xlabel('Axonal conduction delay (ms)')
#     plt.ylabel('Cum. freq.')
#     
# =============================================================================
    delays, counts = calculate_hist_with_fixed_bin_size(delays_rounded, 0.0, 10.0, 0.1)
    
    f = plt.figure()
    plt.step(delays, counts.astype(float) / np.sum(counts))
    plt.xlabel('Axonal conduction delay (ms)')
    plt.ylabel('pdf')
    
    plt.show()
