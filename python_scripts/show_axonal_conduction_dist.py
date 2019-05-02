#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Yevhen Tupikov (2019)

Script visualizes axonal conduction delay distribution
"""
import numpy as np
import matplotlib.pyplot as plt
import struct
import sys, getopt

SIZE_OF_INT = 4
SIZE_OF_DOUBLE = 8

def parseCommandLine():       
    def print_help_message():
        print 'show_axonal_conduction_dist.py -f <file with axonal conduction delays>'
     
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hf:")
    except getopt.GetoptError:
        print_help_message()
        sys.exit(2)
    
    fileArgumentNotPassed = True
    for opt, arg in opts:
        if opt == '-h':
            print_help_message()
            sys.exit()
        elif opt == "-f":
            fileArgumentNotPassed = False
            filename = arg
    if fileArgumentNotPassed:
        print "-f argument is required"
        print_help_message()
        sys.exit(2)
        
    print 'File with axonal delays is', filename
    
    return filename

def read_connections(filename):
    """
    Read neuronal output connections generated from experimental data from binary file
    """
    with open(filename, 'rb') as file:
        data = file.read()
        
        N = struct.unpack("<i", data[:SIZE_OF_INT])[0]
        
        targets_ID = [] # ids of target neurons
        delays = [] # synaptic lengths
        syn_G = [] # synaptic conductances
        
        ind = SIZE_OF_INT    
        
        for i in range(N):
            neurons_ID = struct.unpack("<i", data[ind:(ind+SIZE_OF_INT)])[0]
            number_of_targets = struct.unpack("<i", data[(ind+SIZE_OF_INT):(ind+2*SIZE_OF_INT)])[0]
            
            temp_ID = []
            temp_delay = []
            temp_G = []    
            
            ind += 2*SIZE_OF_INT        
            
            for j in range(number_of_targets):
                
                ID = struct.unpack("<i", data[ind:(ind + SIZE_OF_INT)])[0]
                G = struct.unpack("<d", data[((ind + SIZE_OF_INT)):(ind + SIZE_OF_INT + SIZE_OF_DOUBLE)])[0]
                D = struct.unpack("<d", data[((ind + SIZE_OF_INT + SIZE_OF_DOUBLE)):(ind + SIZE_OF_INT + 2*SIZE_OF_DOUBLE)])[0]
                
                
                temp_ID.append(ID)
                temp_delay.append(D)
                temp_G.append(G)
                
                ind += SIZE_OF_INT + 2*SIZE_OF_DOUBLE
            
            targets_ID.append(temp_ID)
            delays.append(temp_delay)        
            syn_G.append(temp_G)
            
        return (N, targets_ID, syn_G, delays)
    
    
    
if __name__ == "__main__":
    filename = parseCommandLine()
    _, _, _, delays = read_connections(filename)
    
    all_delays = [d for out_delays in delays for d in out_delays]
    
    delays_rounded = np.round(all_delays, decimals=3)
    uniqueDelays, counts = np.unique(delays_rounded, return_counts = True)
    cdf_model = np.cumsum((np.ones(len(uniqueDelays))*counts)/np.float(sum(counts)))   
 
 
    f = plt.figure()
    plt.step(np.concatenate(([0],uniqueDelays)), np.concatenate(([0],cdf_model)), where='post', label='result')
    plt.xlabel('Axonal conduction delay (ms)')
    plt.ylabel('Cum. freq.')

    plt.show()
