#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 23:04:58 2019

@author: eugene

Script reads results of neuron integration time response to input
"""
import struct
import os
import numpy as np
import matplotlib.pyplot as plt

SIZE_OF_INT = 4
SIZE_OF_DOUBLE = 8



def read_results(filename):
    """
    Read results of response to inputs
    """
    with open(filename, "rb") as file:
        data = file.read()
    
        num_trials = struct.unpack("<i", data[0:SIZE_OF_INT])[0] 
        num_spikes_in_trial = np.empty(num_trials, np.int32)
    	
        ind = SIZE_OF_INT
        for i in range(num_trials):
            num_spikes_in_trial[i] = struct.unpack("<i", data[ind:(ind+SIZE_OF_INT)])[0]
            ind += SIZE_OF_INT

        num_trials_with_burst = struct.unpack("<i", data[ind:(ind+SIZE_OF_INT)])[0] 
        burst_onset_times = np.empty(num_trials_with_burst, np.float32)
        
        ind += SIZE_OF_INT
        for i in range(num_trials_with_burst):
            burst_onset_times[i] = struct.unpack("<d", data[ind:(ind+SIZE_OF_DOUBLE)])[0]
            ind += SIZE_OF_DOUBLE
            
        return (num_spikes_in_trial, burst_onset_times)

def combine_capacitance_results(dirname):
    """
    Combine results of simulations with different capacitance that are stored in one directory
    """
    c_d = []
    mean_burst_onset_times = []
    std_burst_onset_times = []
    mean_num_spikes_in_burst = []
    std_num_spikes_in_burst = []
    
    for f in os.listdir(dirname):
        if "cm" in f:
            c_d.append(float(f.split(".bin")[0][2:]))
            
            num_spikes_in_trial, burst_onset_times = read_results(os.path.join(dirname, f))
            
            mean_burst_onset_times.append(np.mean(burst_onset_times))
            std_burst_onset_times.append(np.std(burst_onset_times))
            mean_num_spikes_in_burst.append(np.mean(num_spikes_in_trial))
            std_num_spikes_in_burst.append(np.std(num_spikes_in_trial))
    
    c_d = np.array(c_d)
    mean_burst_onset_times = np.array(mean_burst_onset_times)
    std_burst_onset_times = np.array(std_burst_onset_times)
    mean_num_spikes_in_burst = np.array(mean_num_spikes_in_burst)
    std_num_spikes_in_burst = np.array(std_num_spikes_in_burst)
    
    print c_d
    
    sorted_ind = np.argsort(c_d)
    
    return c_d[sorted_ind], mean_burst_onset_times[sorted_ind], std_burst_onset_times[sorted_ind],\
            mean_num_spikes_in_burst[sorted_ind], std_num_spikes_in_burst[sorted_ind]
 
def read_capacitance_and_integration_times(filename):
    """
    Read dendritic capacitance and integration times from a file
    """    
    with open(filename, "rb") as file:
        data = file.read()
    
        N = struct.unpack("<i", data[0:SIZE_OF_INT])[0] 
        a = struct.unpack("<{0}d".format(N*2), data[SIZE_OF_INT:])
        
        capacitance_dend = a[::2]
        integration_times = a[1::2]
        
        capacitance_dend = np.array(capacitance_dend)
        integration_times = np.array(integration_times)
       
    return capacitance_dend, integration_times
    
def plot_capacitance_and_integration_time_model(filename):
    """
    Plots dendritic capacitance and putative integration time of neurons in the model
    """
    capacitance_dend, integration_times = read_capacitance_and_integration_times(filename)

    sorted_ind = np.argsort(capacitance_dend)
        
    capacitance_dend = capacitance_dend[sorted_ind]
    integration_times = integration_times[sorted_ind]
       
    
    print "Mean integration time: ",np.mean(integration_times)
    print "Std integration time: ",np.std(integration_times)
    print "Mean capacitance: ",np.mean(capacitance_dend)
    print "Std capacitance: ",np.std(capacitance_dend)
    
    plt.figure()
    plt.plot(capacitance_dend, integration_times, '-o')
    plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
    plt.ylabel('Putative integration time (ms)')
    
    plt.figure()
    plt.hist(capacitance_dend, bins=100)
    plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
    plt.ylabel('Count')
    
    plt.figure()
    plt.hist(integration_times, bins=100)
    plt.xlabel('integration time (ms)')
    plt.ylabel('Count')
    
    plt.show()

def read_hh2(filename):
    """
    Reads output of HH2 neuron
    """
    with open(filename, mode = "rb") as f:
        data = f.read()
        
        # calculate number of datapoints
        num_datapoints = len(data) / (7 * SIZE_OF_DOUBLE)
        
        t = np.empty(num_datapoints, np.float32)
        Vs = np.empty(num_datapoints, np.float32)
        Vd = np.empty(num_datapoints, np.float32)
        Gexc_d = np.empty(num_datapoints, np.float32)
        Ginh_d = np.empty(num_datapoints, np.float32)
        Is = np.empty(num_datapoints, np.float32)
        Id = np.empty(num_datapoints, np.float32)
        
        a = np.array(struct.unpack("<{0}d".format(num_datapoints*7), data))
        
        t = a[::7]
        Vs = a[1::7]
        Vd = a[2::7]
        Gexc_d = a[3::7]
        Ginh_d = a[4::7]
        Is = a[5::7]
        Id = a[6::7]
        
        return t, Vs, Vd, Gexc_d, Ginh_d, Is, Id

def estimate_noise(filename):
    """
    Calculate the amplitude of membrane potential fluctuations
    """
    t, Vs, _, _, _, _, _ = read_hh2(filename)
    
    print "mean V = ",np.mean(Vs)
    print "std V = ",np.std(Vs)
    
    fired = np.zeros_like(Vs, np.int8)
    fired[np.where(Vs >= 0)] = 1
    
    num_spikes = len(np.where(np.diff(fired) == 1)[0])
    
    print num_spikes
    print "Spike frequency = ",float(num_spikes) * 1000.0 / t[-1]
    
def plot_noise_vs_dend_capacitance_model(file_noise, file_capacitance):
    """
    Plot noise strength versus dendritic capacitance in the model
    """
    capacitance_dend, _ = read_capacitance_and_integration_times(file_capacitance)

    
    with open(file_noise, mode = "rb") as f:
        data = f.read()
        
        N = struct.unpack("<i", data[:SIZE_OF_INT])[0]
        
        a = np.array(struct.unpack("<{0}d".format(N*4), data[SIZE_OF_INT:]))
        
        mu_soma = a[::4]
        std_soma = a[1::4]
        mu_dend = a[2::4]
        std_dend = a[3::4]
        
        
    
    plt.figure()
    plt.scatter(capacitance_dend, mu_soma)
    plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
    plt.ylabel('mu_soma (nA)')
  
    plt.figure()
    plt.scatter(capacitance_dend, std_soma)
    plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
    plt.ylabel('std_soma (nA)')
  
    plt.figure()
    plt.scatter(capacitance_dend, mu_dend)
    plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
    plt.ylabel('mu_dend (nA)')
  
    plt.figure()
    plt.scatter(capacitance_dend, std_dend)
    plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
    plt.ylabel('std_dend (nA)')
  
    plt.show()
        


def plot_noise_vs_dend_capacitance(dirname):
    """
    Plot voltage fluctuations versus dendritic capacitance
    """
    c_d = []
    std_V = []
    
    for f in os.listdir(dirname):
        c_d.append(float(f.split(".bin")[0][2:]))
        print c_d[-1]
        _, Vs, _, _, _, _, _ = read_hh2(os.path.join(dirname, f))
        #print t
        #print Vs
        
        std_V.append(np.std(Vs))
    
    c_d, std_V = zip(*sorted(zip(c_d, std_V)))
    
    plt.figure()
    plt.plot(c_d, std_V, '-o')
    plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
    plt.ylabel('Std V (mV)')  
    plt.show()
        
            
def plot_capacitance_tuning_curve(dirname):
    """
    Plot neuronal response properties with different dendritic capacitance
    """
    c_d, mean_burst_onset_times, std_burst_onset_times, \
            mean_num_spikes_in_burst, std_num_spikes_in_burst = combine_capacitance_results(dirname)
    
    plt.figure()
    plt.plot(c_d, mean_burst_onset_times, '-o')
    plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
    plt.ylabel('Integration time (ms)')
    
    plt.figure()
    plt.plot(c_d, std_burst_onset_times, '-o')
    plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
    plt.ylabel('Std integration time (ms)')
    
    plt.figure()
    plt.plot(c_d, mean_num_spikes_in_burst, '-o')
    plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
    plt.ylabel('Mean # spikes in burst')
    
    plt.figure()
    plt.plot(c_d, std_num_spikes_in_burst, '-o')
    plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
    plt.ylabel('Std # spikes in burst')
    
    plt.show()
   
if __name__ == "__main__":
# =============================================================================
#     ### Response to input at different dendritic capacitance ###
#     #filename = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/integrationTime/sm1.0/cm3.8.bin"
#     #filename = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/integrationTime/smallCmSmallA/cm0.5_dt0.01.bin"
#     filename = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/integrationTime/sm6.0/cm0.5.bin"
#     
#     num_spikes_in_trial, burst_onset_times = read_results(filename)
#     
#     print num_spikes_in_trial
#     print burst_onset_times
#     
#     if len(num_spikes_in_trial) != len(burst_onset_times):
#         print "Burst robustness: ",float(len(num_spikes_in_trial[num_spikes_in_trial>0])) / float(len(num_spikes_in_trial))
#         
#     print "Mean number of spikes: ",np.mean(num_spikes_in_trial[num_spikes_in_trial>0])
#     print "Std number of spikes: ",np.std(num_spikes_in_trial[num_spikes_in_trial>0])
#     
#     print "Mean burst onset time: ",np.mean(burst_onset_times)
#     print "Std burst onset time: ",np.std(burst_onset_times)
#     
#     ### comparison of integration time vs dendritic capacitance graphs for different width of synchronization windows
#     dirname = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/integrationTime/sm8.0/"
#     
#     c_d, mean_burst_onset_times, std_burst_onset_times, \
#             mean_num_spikes_in_burst, std_num_spikes_in_burst = combine_capacitance_results(dirname)
#     
#     from scipy import stats
#     slope, intercept, r_value, p_value, std_err = stats.linregress(c_d, mean_burst_onset_times)
#     
#     
#     dirname_2 = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/integrationTime/sm6.0/"
#     
#     c_d_2, mean_burst_onset_times_2, std_burst_onset_times_2, \
#             mean_num_spikes_in_burst_2, std_num_spikes_in_burst_2 = combine_capacitance_results(dirname_2)
#     
#     
#     plt.figure()
#     plt.plot(c_d, mean_burst_onset_times, '-o', label='sm 8.0')
#     plt.plot(c_d_2, mean_burst_onset_times_2, '-o', label='sm 6.0')
#     
#     #plt.plot(c_d, intercept + slope*c_d, 'r')
#     plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
#     plt.ylabel('Mean burst onset time (ms)')
#     plt.legend()
#     
#     plt.figure()
#     plt.plot(c_d, std_burst_onset_times, '-o', label='sm 1.0')
#     plt.plot(c_d_2, std_burst_onset_times_2, '-o', label='sm 6.0')
#     plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
#     plt.ylabel('Std burst onset time (ms)')
#     plt.legend()
#    
#     plt.figure()
#     plt.plot(c_d, mean_num_spikes_in_burst, '-o', label='sm 1.0')
#     plt.plot(c_d_2, mean_num_spikes_in_burst_2, '-o', label='sm 6.0')
#     plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
#     plt.ylabel('Mean # spikes in burst')
#     plt.legend()
#     
#     plt.figure()
#     plt.plot(c_d, std_num_spikes_in_burst, '-o', label='sm 1.0')
#     plt.plot(c_d_2, std_num_spikes_in_burst_2, '-o', label='sm 6.0')
#     plt.xlabel('Capacitance dendrite ($\mu F/ cm^2$)')
#     plt.ylabel('Std # spikes in burst')
#     plt.legend()
#     
# =============================================================================
    
    
    #plot_capacitance_tuning_curve("/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/integrationTime/sm6.0/")
    
    plot_capacitance_and_integration_time_model("/home/eugene/Programming/data/mlong/integrationConst/poly15/cm_dend_and_integration_times.bin")
    
    #plt.show()
   
    #dirname = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/noise_s0.1_d0.2/"    
    #plot_noise_vs_dend_capacitance(dirname)

    #filename = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/cm1.0/noise_s0.1_d0.198.bin"
    #filename = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/noise_s0.1_d0.2/cm1.0.bin"
    
    #estimate_noise(filename)
    
    
    #file_noise = "/home/eugene/Programming/data/mlong/integrationConst/poly2/test_same_noise/noise.bin"
    #file_capacitance = "/home/eugene/Programming/data/mlong/integrationConst/poly2/cm_dend_and_integration_times.bin"
    #plot_noise_vs_dend_capacitance_model(file_noise, file_capacitance)

    #dirname = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/same_noise/"
    #plot_noise_vs_dend_capacitance(dirname)
    
    #filename = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/integrationTime/smallCm/cm0.5.bin"
    #num_spikes_in_trial, burst_onset_times = read_results(filename)
    
    #print num_spikes_in_trial
    #print burst_onset_times
    
    
    #plot_capacitance_and_integration_time_model("/home/eugene/Programming/data/mlong/integrationConst/poly3/cm_dend_and_integration_times.bin")
#     
    #estimate_noise("/home/eugene/Programming/data/mlong/noise/noiseCheckDebrabandNew/noise_s0.30_d0.0_dt0.02.bin")