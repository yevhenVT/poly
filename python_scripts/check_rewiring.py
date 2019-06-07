#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 23:19:46 2019

@author: eugene

Script checks that rewiring procedure of the network was performed correctly
"""
from show_axonal_conduction_dist import read_connections
import numpy as np
import matplotlib.pyplot as plt

def plot_indOut(file_connections):
    """
    Plot in and out degrees for neurons in the network
    """
    (N, targets_ID, _, _) = read_connections(file_connections)
    
    out_degrees = np.empty(N, np.int32)
    in_degrees = np.zeros(N, np.int32)
    
    for i in range(N):
        out_degrees[i] = len(targets_ID[i])
        
        for target in targets_ID[i]:
            in_degrees[target] += 1
    
    print np.where(in_degrees > 300)[0]
    
    plt.figure()
    plt.hist(out_degrees, bins=30)
    plt.xlabel('Out degree')
    plt.ylabel('Count')
    
    plt.figure()
    plt.hist(in_degrees, bins=30)
    plt.xlabel('In degree')
    plt.ylabel('Count')
    
    plt.show()
    

def find_fraction_rewired(file_connections_original, file_connections_rewired):
    """
    Calculates the fraction of rewired connections by comparing connections in the original network
    to connections in the rewired one
    """
    (N, targets_ID_original, _, _) = read_connections(file_connections_original)
    (_, targets_ID_rewired, _, _) = read_connections(file_connections_rewired)
    
    num_connections_total = 0
    num_connections_rewired = 0
    
    for i in range(N):
        assert len(targets_ID_original[i]) == len(targets_ID_rewired[i])
        num_connections_total += len(targets_ID_original[i])
        
        for j in range(len(targets_ID_original[i])):
            if targets_ID_original[i][j] != targets_ID_rewired[i][j]:
                num_connections_rewired += 1
                
    return float(num_connections_rewired) / float(num_connections_total)
    

if __name__ == "__main__":
    file_connections_original = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/RA_RA_connections.bin"
    file_connections_rewired = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.7/RA_RA_connections.bin"
    
    fraction_rewired = find_fraction_rewired(file_connections_original, file_connections_rewired)
    
    print fraction_rewired

    plot_indOut(file_connections_rewired)
