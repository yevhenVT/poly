#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Yevhen Tupikov

Script plots cdf of axonal conduction delays for grown polychronous networks
"""
import matplotlib.pyplot as plt
import getopt, sys
import os
from show_spike_raster import plot_spikes

def parseCommandLine(): 
    def print_help_message():
        print 'show_results_polychronous.py -m <mean delay> -s <std delay> -d <time duration to plot>'
 
    duration = 100.0
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hm:s:d:")
    except getopt.GetoptError:
        print_help_message()
        sys.exit(2)
    
    meanArgumentNotPassed = True
    stdArgumentNotPassed = True
    
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt == "-m":
            meanArgumentNotPassed = False
            mean = float(arg)
        elif opt == "-s":
            stdArgumentNotPassed = False
            std = float(arg)
        elif opt == "-d":
            duration = float(arg)
       
    if meanArgumentNotPassed:
        print "-m argument is required"
        print_help_message()          
        sys.exit(2)
    
    if stdArgumentNotPassed:
        print "-s argument is required"
        print_help_message()           
        sys.exit(2)
    
    
    print 'Mean axonal delay is', mean
    print 'Std axonal delay is', std
    print 'Time duration to plot is', duration
    
    return mean, std, duration

def read_cdf(filename):
    """
    Reads an axonal conduction delay from txt file
    """
    cdf = []
    delay = []
    
    with open(filename, 'r') as f:
        for l in f:
            d, c = l.strip().split('\t')
            cdf.append(float(c))
            delay.append(float(d))
            
    return delay, cdf
            
if __name__ == "__main__":
    mean, std, duration = parseCommandLine()
    
    script_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    #print"{0.2f}".format(123.253434)
    
    delay, cdf = read_cdf(os.path.join(script_parent_dir, "data/AxonalDelays/mean{0:.1f}ms_std{1:.2f}ms_cdf_axonalDelays.txt".format(mean, std)))
   
    plt.figure()
    plt.step(delay, cdf)
    
    plt.xlabel('Axonal conduction delay (ms)')
    plt.ylabel('Cum. freq')
    plt.show()

    
    
    plot_spikes(duration, os.path.join(script_parent_dir, "data/Spikes/mean{0:.1f}ms_std{1:.2f}ms_trial0_somaSpikes.bin".format(mean, std)))
    plt.show()
    
    
    


