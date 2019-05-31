#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:25:10 2019

@author: eugene
"""
from show_spike_raster import read_spikes
from show_spike_raster import plot_spikes

from show_axonal_conduction_dist import read_connections

import os
import numpy as np
import matplotlib.pyplot as plt

BURST_DURATION = 20.0


def calculate_average_burst_onset_density(simDir, netDir, start=-5.0, end=1000.0, bin_width=0.75, min_num_spikes_in_burst=2, max_interspike_interval=10.0):
    """
    Calculate average burst onset density of connected neurons
    """
    num_trials = 0
    
    N, connected_neurons = find_connected_neurons(netDir)
    
    size = int((end - start) / bin_width) + 1
    density_all_trials = np.zeros(size, np.int32)    
    
    for f in os.listdir(simDir):
        if "somaSpikes" in f:  
            num_trials += 1
            
            burst_onsets, _ = find_burst_onsets(os.path.join(simDir, f), min_num_spikes_in_burst, max_interspike_interval)
            
            all_burst_onsets = sorted([burst for n in connected_neurons for burst in burst_onsets[n]])
            
            time, density = calculate_hist_with_fixed_bin_size(all_burst_onsets, start, end, bin_width)
            density_all_trials += density
            
    density_all_trials = density_all_trials / float(num_trials)
    
    plt.figure()
    plt.plot(time, density_all_trials / bin_width)
    plt.xlabel('Time (ms)')
    plt.ylabel('Average burst density (1/ms)')
    
    plt.ylim([0, np.max(density_all_trials / bin_width) + 5])
    
    plt.figure()
    plt.plot(time, density_all_trials / bin_width)
    plt.xlabel('Time (ms)')
    plt.ylabel('Average burst density (1/ms)')
    plt.ylim([0, np.max(density / bin_width)+5])
    plt.xlim([50, 200])

    
    plt.show()
    
    return time, density_all_trials / bin_width


def write_to_file(x, y, filename):
    """
    Write 2 arrays to a txt file
    """
    with open(filename, 'w') as f:
        for xx, yy in zip(x, y):
            f.write(str(xx))
            f.write("\t")
            f.write(str(yy))
            f.write("\n")

def plot_burst_onset_density(filename, start=-5.0, end=1000.0, bin_width=0.75, min_num_spikes_in_burst=2, max_interspike_interval=10.0):
    """
    Plot density of burst onset times
    """
    burst_onsets, _ = find_burst_onsets(filename, min_num_spikes_in_burst, max_interspike_interval)

    all_burst_onsets = sorted([burst for bursts in burst_onsets for burst in bursts])
    
    time, density = calculate_hist_with_fixed_bin_size(all_burst_onsets, start, end, bin_width)
    
    
    plt.figure()
    plt.plot(time, density / bin_width)
    plt.xlabel('Time (ms)')
    plt.ylabel('Burst density (1/ms)')
    
    
    plt.show()

def calculate_hist_with_fixed_bin_size(a, start, end, bin_width):
    """
    Calculate histogram on array a with fixed size bin_width in range [start, end]
    """
    size = int((end - start) / bin_width) + 1
    x = np.array([float(i)*bin_width + start + bin_width/2. for i in range(size)])
    
    counts = np.zeros(size, np.int32)    
    
    #print min(burst_times)
    
    a = map(lambda x: x - start, a)
    a = sorted(a)
    #print min(burst_times)
    
    for aa in a:
        ind = int(aa / bin_width)
        if ind > size - 1:
            continue
        else:
            counts[ind] += 1
        
    return x, counts


def find_connected_neurons(netDir):
    """
    Find all neurons that either send outputs or receive inputs
    """
    (N, targets_ID, _, _) = read_connections(os.path.join(netDir, "RA_RA_connections.bin"))
    
    connected_neurons = set()
    
    for i in range(N):
        if len(targets_ID[i]) > 0:
            connected_neurons.add(i)
        for target in targets_ID[i]:
            connected_neurons.add(target)
            
    return N, list(connected_neurons)
    

def analyze_dendritic_spikes_multiple_trials(simDir, netDir):
    """
    Calculate average densritic spike time and its jitter based on multiple files with dendritic spike info
    """
    N, connected_neurons = find_connected_neurons(netDir)
    
    dendritic_spikes = [[] for _ in range(N)]
    
    num_trials = 0

    for f in os.listdir(simDir):
        if "dendSpikes" in f:  
            num_trials += 1
    
            (_, dend_spike_times_raw, neuron_fired) = read_spikes(os.path.join(simDir, f))

            for spike_times, nids in zip(dend_spike_times_raw, neuron_fired):
                dendritic_spikes[nids[0]].extend(spike_times)
                if len(spike_times) > 1:
                    print "Neuron {0} has {1} dendritic spikes".format(nids[0], len(spike_times))
                    
                    
       
    mean_dendritic_spike_times = np.empty(N, np.float32)
    mean_dendritic_spike_times.fill(-1.0)
    
    std_dendritic_spike_times = np.empty(N, np.float32)
    std_dendritic_spike_times.fill(-1.0)
    
    for i in range(N):
        if len(dendritic_spikes[i]) > 0:
            mean_dendritic_spike_times[i] = np.mean(dendritic_spikes[i])
            std_dendritic_spike_times[i] = np.std(dendritic_spikes[i])
                
    
    
    print "Neurons with large jitter: ",np.where(std_dendritic_spike_times > 1.0)[0]
    
    std_dendritic_spike_times_connected = std_dendritic_spike_times[connected_neurons]
    mean_dendritic_spike_times_connected = mean_dendritic_spike_times[connected_neurons]
    
    mean_dendritic_spike_times_connected = mean_dendritic_spike_times_connected - np.min(mean_dendritic_spike_times_connected[mean_dendritic_spike_times_connected > 0])
    
    
    jitter_bursted_connected = std_dendritic_spike_times_connected[std_dendritic_spike_times_connected >= 0]
    dendritic_time_bursted_connected = mean_dendritic_spike_times_connected[mean_dendritic_spike_times_connected >= 0]
    
    print "Mean jitter: ",np.mean(jitter_bursted_connected)
    print "Std jitter: ",np.std(jitter_bursted_connected)
    
    
    time, counts = calculate_hist_with_fixed_bin_size(jitter_bursted_connected, 0.0, 20.0, 0.02)
        
    plt.figure()
    plt.step(time, counts)
    plt.xlabel('Jitter in dendritic spikes (ms)')
    plt.ylabel('Count')
    
    
    bin_width = 0.75
    time, burst_density = calculate_hist_with_fixed_bin_size(np.sort(dendritic_time_bursted_connected), -10.0, 2000.0, bin_width)

    #write_to_file(time, burst_density, file_burst_density)
    
    plt.figure()
    
    plt.plot(time, burst_density)
    plt.xlabel('Time (ms)')
    plt.ylabel('Dendritic spike density (1/ms)')
    
    plt.show()
    
    
def find_burst_onsets(filename, min_num_spikes_in_burst, max_interspike_interval):
    """
    Find burst onset times of neurons in a given simulation.
    Burst is defined as a sequence of minimum min_num_spikes_in_burst spikes
    with max interspike interval max_interspike_interval
    """
    (N, spike_times_raw, neuron_fired) = read_spikes(os.path.join(simDir, filename))

    burst_onsets = [[] for _ in range(N)]
    num_spikes_in_burst = [[] for _ in range(N)]
        
    for spike_times, nids in zip(spike_times_raw, neuron_fired):
        if len(spike_times) > 0:          
            prevBurstOnset = spike_times[0]
            prevSpikeInBurst = spike_times[0]
            numSpikesInBurst = 1
            
            for spike in spike_times[1:]:
                if spike - prevSpikeInBurst <= max_interspike_interval:
                    prevSpikeInBurst = spike
                    numSpikesInBurst += 1
                else:
                    if numSpikesInBurst >= min_num_spikes_in_burst:
                        num_spikes_in_burst[nids[0]].append(numSpikesInBurst)
                        burst_onsets[nids[0]].append(prevBurstOnset)
                       
                    prevBurstOnset = spike
                    prevSpikeInBurst = spike
                    numSpikesInBurst = 1
           
            if numSpikesInBurst >= min_num_spikes_in_burst:
                num_spikes_in_burst[nids[0]].append(numSpikesInBurst)
                burst_onsets[nids[0]].append(prevBurstOnset)
                              
                
    return burst_onsets, num_spikes_in_burst
            
           
def analyze_spikes_multiple_trials(simDir, netDir, fileJitter=None):
    """
    Calculate average first spike time and jitter based on multiple files with spike info
    """
    N, connected_neurons = find_connected_neurons(netDir)
    
    ### burst is defined as >= 2 spikes with separation <= 10 ms between spikes
    burst_onsets_trial = [[] for _ in range(N)]
    num_spikes_in_burst_trial = [[] for _ in range(N)]
    
    INTERSPIKE_INTERVAL = 10.0
    MIN_NUM_SPIKES_IN_BURST = 2
    
    num_trials = 0
    
    for f in os.listdir(simDir):
        if "somaSpikes" in f:  
            num_trials += 1
            burst_onsets_one_trial, num_spikes_in_burst_one_trial = find_burst_onsets(os.path.join(simDir, f), MIN_NUM_SPIKES_IN_BURST, INTERSPIKE_INTERVAL)          
            
            for i, (b_o, n_s) in enumerate(zip(burst_onsets_one_trial, num_spikes_in_burst_one_trial)):
                if len(b_o) > 0:
                    burst_onsets_trial[i].append(b_o)
                    num_spikes_in_burst_trial[i].append(n_s)
    
    print burst_onsets_trial[0]
    
    burst_onsets_allTrials = [[] for _ in range(N)] 
    num_spikes_in_burst_allTrials = [[] for _ in range(N)] 
    
    for i in range(N):
        burst_onsets_allTrials[i] = [item for sublist in burst_onsets_trial[i] for item in sublist]
        num_spikes_in_burst_allTrials[i] = [item for sublist in num_spikes_in_burst_trial[i] for item in sublist]
    
    
    median_burst_onset_time = np.empty(N, np.float32)
    iqr_burst_onset_time = np.empty(N, np.float32)
    
    for n in connected_neurons:
        if len(burst_onsets_allTrials[n]) > 0:
            median_burst_onset_time[n] = np.median(burst_onsets_allTrials[n])
            iqr_burst_onset_time[n] = np.subtract(*np.percentile(burst_onsets_allTrials[n], [75, 25]))

        else:
            median_burst_onset_time[n] = -1.0
            iqr_burst_onset_time[n] = -1.0
    
    mean_burst_onset_time = np.empty(N, np.float32)
    std_burst_onset_time = np.empty(N, np.float32)
    
    print median_burst_onset_time[connected_neurons]
    print iqr_burst_onset_time[connected_neurons]
    
    
    for n in connected_neurons:
        if median_burst_onset_time[n] > 0:
            relevantBursts = []
            
            for burst in burst_onsets_allTrials[n]:
                if (burst >= median_burst_onset_time[n]-1.5*iqr_burst_onset_time[n]) and (burst <= median_burst_onset_time[n]+1.5*iqr_burst_onset_time[n]):
                   relevantBursts.append(burst)
                   
            if len(relevantBursts) > 0:
                mean_burst_onset_time[n] = np.mean(relevantBursts)
                std_burst_onset_time[n] = np.std(relevantBursts)
            else:
                mean_burst_onset_time[n] = -1.0
                std_burst_onset_time[n] = -1.0
        
        else:
            mean_burst_onset_time[n] = -1.0
            std_burst_onset_time[n] = -1.0
    
    jitter_bursted_connected = [std_burst_onset_time[n] for n in connected_neurons if std_burst_onset_time[n] >= 0]
    
    print "Mean jitter: ",np.mean(jitter_bursted_connected)
    print "Std jitter: ",np.std(jitter_bursted_connected)
    
    time, counts = calculate_hist_with_fixed_bin_size(jitter_bursted_connected, 0.0, 20.0, 0.02)    
    
    if fileJitter is not None:
        write_to_file(time, counts, fileJitter)
    
    plt.figure()
    plt.step(time, counts)
    plt.xlabel('Jitter (ms)')
    plt.ylabel('Count')
    plt.xlim([0, 1.5])
    plt.ylim([0, np.max(counts)+50])

    plt.show()  
    
    
# =============================================================================
#     numSpikesInBurstWhenSingleBurst = []
#     maxNumSpikesInBurstWhenMultipleBurst = []
#     
#     for n in connected_neurons:
#         for i in range(len(burst_onsets_trial[n])):
#             if len(burst_onsets_trial[n][i]) > 1:
#                 maxNumSpikes = -1
#                 
#                 for b, num in zip(burst_onsets_trial[n][i], num_spikes_in_burst_trial[n][i]):
#                     if num > maxNumSpikes:
#                         maxNumSpikes = num
#                         
#                 maxNumSpikesInBurstWhenMultipleBurst.append(maxNumSpikes)
#                 
#             elif len(burst_onsets_trial[n][i]) == 1:
#                 numSpikesInBurstWhenSingleBurst.append(num_spikes_in_burst_trial[n][i][0])
#     
#     plt.figure()
#     plt.hist(numSpikesInBurstWhenSingleBurst)
#     plt.xlabel('# spikes in burst when there is only one')
#     plt.ylabel('Count')
#    
#     plt.figure()
#     plt.hist(maxNumSpikesInBurstWhenMultipleBurst)
#     plt.xlabel('# spikes in burst when there are multiple')
#     plt.ylabel('Count')
#    
#     
#     
#     mean_burst_onset_times = np.empty(N, np.float32)
#     mean_burst_onset_times.fill(-1.0)
#     
#     std_burst_onset_times = np.empty(N, np.float32)
#     std_burst_onset_times.fill(-1.0)
#     
#     
#     for i in range(N):
#         if len(burst_onsets_allTrials[i]) > 0:     
#             mean_burst_onset_times[i] = np.mean(burst_onsets_allTrials[i])
#             std_burst_onset_times[i] = np.std(burst_onsets_allTrials[i])
#                 
#     
#     
#     print "Neurons with large jitter: ",np.where(std_burst_onset_times > 1.0)[0]
#     #print burst_onsets[70], len(burst_onsets[70])
#     #print burst_onsets[116], len(burst_onsets[116])
#     #print burst_onsets[272], len(burst_onsets[272])
#     
#     std_burst_onset_times_connected = std_burst_onset_times[connected_neurons]
#     mean_burst_onset_times_connected = mean_burst_onset_times[connected_neurons]
#     mean_burst_onset_times_connected = mean_burst_onset_times_connected - np.min(mean_burst_onset_times_connected[mean_burst_onset_times_connected > 0])
#     
#     
#     jitter_bursted_connected = std_burst_onset_times_connected[std_burst_onset_times_connected >= 0]
#     burst_time_bursted_connected = mean_burst_onset_times_connected[mean_burst_onset_times_connected >= 0]
#     
#     print "Mean jitter: ",np.mean(jitter_bursted_connected)
#     print "Std jitter: ",np.std(jitter_bursted_connected)
#     
#     meanNumBurst = np.zeros(N, np.float32)
#     
#     for i, burst_onset_times in enumerate(burst_onsets_allTrials):
#         if len(burst_onset_times) > 0:
#             meanNumBurst[i] = float(len(burst_onset_times)) / float(num_trials)
#             
#     
#     plt.figure()
#     plt.hist([item for sublist in num_spikes_in_burst_allTrials for item in sublist])
#     plt.xlabel('# spikes in burst')
#     plt.ylabel('Count')
#    
#     
#     plt.figure()
#     plt.hist(burst_onsets_allTrials[0])
#     plt.xlabel('Burst onset times for neuron 0')
#     plt.ylabel('Count')
#    
#     
#     plt.figure()
#     plt.hist(meanNumBurst[connected_neurons])
#     plt.xlabel('mean # bursts per neuron')
#     plt.ylabel('Count')
#     
#     
#     
#     time, counts = calculate_hist_with_fixed_bin_size(jitter_bursted_connected, 0.0, 20.0, 0.02)
#         
#     plt.figure()
#     plt.step(time, counts)
#     plt.xlabel('Jitter (ms)')
#     plt.ylabel('Count')
#     
#     
#     bin_width = 0.75
#     time, burst_density = calculate_hist_with_fixed_bin_size(np.sort(burst_time_bursted_connected), -10.0, 2000.0, bin_width)
# 
#     #write_to_file(time, burst_density, file_burst_density)
#     
#     plt.figure()
#     
#     plt.plot(time, burst_density)
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Burst density (1/ms)')
#     
#     plt.show()
#             #print trial_number, simulation_time
# =============================================================================


if __name__ == "__main__":
    simDir = "/home/eugene/Programming/data/mlong/integrationConst/poly2/test_same_noise/"
    netDir = "/home/eugene/Programming/data/mlong/integrationConst/poly2/"
    
    
    
    #simDir = "/home/eugene/Programming/data/mlong/noise/052519/noise_s0.25_d0.0/"
    #netDir = "/home/eugene/Programming/data/mlong/noise/network/"
    
    #file_jitter = "/home/eugene/Programming/figures/mlong/noise/052519/jitter_relevant_bursts_stdIs0.25.txt"
    #file_burst_density = "/home/eugene/Programming/figures/mlong/noise/052519/average_burst_density_stdIs0.10.txt"
    
    #analyze_spikes_multiple_trials(simDir, netDir, file_jitter)
    analyze_spikes_multiple_trials(simDir, netDir)
    
    #analyze_dendritic_spikes_multiple_trials(simDir, netDir)
    
    #plot_spikes(300.0, "/home/eugene/Programming/data/mlong/noise/052519/noise_s0.20_d0.0/testTrial_0_somaSpikes.bin")
    #plot_spikes(300.0, "/home/eugene/Programming/data/mlong/noise/052519/noise_s0.25_d0.0/testTrial_0_dendSpikes.bin")
    #plot_burst_onset_density("/home/eugene/Programming/data/mlong/integrationConst/poly2/e0.004000_i0.000000_somaSpikes.bin")
    #plot_burst_onset_density("/home/eugene/Programming/data/mlong/integrationConst/poly2/test_same_cm/testTrial_0_somaSpikes.bin")
    
    #time, average_burst_density = calculate_average_burst_onset_density(simDir, netDir)
    #write_to_file(time, average_burst_density, file_burst_density)
    
    
# =============================================================================
#     simDir = "/home/eugene/Programming/data/mlong/integrationConst/poly2/test/"
#     netDir = "/home/eugene/Programming/data/mlong/integrationConst/poly2/"
#     
#     time_2, average_burst_density_2 = calculate_average_burst_onset_density(simDir, netDir)
#     
#     plt.figure()
#     plt.plot(time, average_burst_density, label='same cm')
#     plt.plot(time_2, average_burst_density_2, label='different cm')
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Average burst density (1/ms)')
#     plt.legend()
#     plt.show()
# =============================================================================
    