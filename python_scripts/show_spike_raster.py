#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Yevhen Tupikov (2019)

Script shows spike raster plot
"""

import matplotlib.pyplot as plt
import struct

import sys, getopt

SIZE_OF_INT = 4
SIZE_OF_DOUBLE = 8

def plot_spikes(duration, filename):
    (N, spike_times_raw, neuron_fired) = read_spikes(filename)
    
    if len(spike_times_raw) == 0:
        print "No spikes!"
        sys.exit()
    
    spike_times = sorted(spike_times_raw)
    spike_times = [[spike - spike_times[0][0] for spike in spikes] for spikes in spike_times]
    
    
    f = plt.figure()
    ax1 = f.add_subplot(111)
    
    exceededDuration = False

    # show 10% of spikes
    for i in range(len(spike_times)):
        if i % 10 == 0:
            for spike in spike_times[i]:
                ax1.vlines(spike, i-0.5, i+0.5)
            
                if spike > duration:
                    exceededDuration = True
        if exceededDuration:
            break
    
    plt.tick_params(axis='y',which='both',bottom='off',top='off',labelbottom='off')
        
    
    ax1.set_ylabel("Neuron id")
    ax1.set_xlabel("Time (ms)")
    ax1.set_xlim([0, duration])
    plt.show()    

def read_spikes(filename):
    with open(filename, "rb") as file:
        data = file.read()
    
        N = struct.unpack("<i", data[0:SIZE_OF_INT])[0] 
        ind = SIZE_OF_INT
        spike_times = []
        neuron_fired = []
    		
        for i in range(N):
            spike_array_size = struct.unpack("<i", data[ind:(ind+SIZE_OF_INT)])[0]
            single_neuron_spikes = []        
            single_neuron_fired = []        
            ind += SIZE_OF_INT
            for j in range(spike_array_size):
                temp = struct.unpack("<d", data[ind:(ind + SIZE_OF_DOUBLE)])[0]
                single_neuron_spikes.append(temp)
                single_neuron_fired.append(i)
                ind += SIZE_OF_DOUBLE
            			
            if len(single_neuron_spikes) > 0:
               spike_times.append(single_neuron_spikes)
               neuron_fired.append(single_neuron_fired)

    return N, spike_times, neuron_fired
		

def parseCommandLine(): 
    def print_help_message():
        print 'show_spike_raster.py -f <file with spike info> -d <time duration to plot>'      
    
    duration = 150.0
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hf:d:")
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
        elif opt == "-d":
            duration = float(arg)
    if fileArgumentNotPassed:
        print "-f argument is required"
        print_help_message()
        sys.exit(2)
        
    print 'File with spikes is', filename
    print 'Time duration to show is', duration
    
    return duration, filename


if __name__ == "__main__":
    duration, filename = parseCommandLine()
   
    plot_spikes(duration, filename)
