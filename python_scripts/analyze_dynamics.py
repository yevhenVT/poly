#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:25:10 2019

@author: eugene
"""
from show_spike_raster import read_spikes
from show_spike_raster import plot_spikes
from analyze_input_response import read_hh2
from show_axonal_conduction_dist import read_connections

from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt

BURST_DURATION = 20.0


def plot_ordered_noisy_spikes(duration, file_spikes, file_order):
    """
    Script plots somatic spikes from file file_spikes according to the order in which
    neuron spiked in file file_order
    """
    (N, spike_times_raw, neuron_fired) = read_spikes(file_spikes)
    (_, spike_times_raw_order, neuron_fired_order) = read_spikes(file_order)
    
    
    spike_times = [[] for _ in range(N)]
    
    for spikes, n in zip(spike_times_raw, neuron_fired):
        spike_times[n[0]] = spikes
    
    spike_times_raw_order, neuron_fired_order = zip(*sorted(zip(spike_times_raw_order, neuron_fired_order)))
    
    order = [n[0] for n in neuron_fired_order]
    
    f = plt.figure()
    ax1 = f.add_subplot(111)
    
    # show 10% of spikes
    for counter, i in enumerate(order):
        if counter % 10 == 0:
            for spike in spike_times[i]:
                if spike < duration:
                    ax1.vlines(spike, counter-0.5, counter+0.5)
            
                
    plt.tick_params(axis='y',which='both',bottom='off',top='off',labelbottom='off')
        
    
    ax1.set_ylabel("Neuron id")
    ax1.set_xlabel("Time (ms)")
    ax1.set_xlim([0, duration])

    plt.show()    
    
def plot_input_times_low_in_degree(file_connections, file_spikes):
    """
    Plot input times of neurons with low in-degree that spike close to starters
    """
    
    INTERSPIKE_INTERVAL = 10.0
    MIN_NUM_SPIKES_IN_BURST = 2
        
    burst_onsets_one_trial, num_spikes_in_burst_one_trial = find_burst_onsets(file_spikes, MIN_NUM_SPIKES_IN_BURST, INTERSPIKE_INTERVAL)          
    (N, targets_ID, _, delays) = read_connections(file_connections)
   
    burst_onsets = np.empty(N, np.float32)
    for i in range(N):
        if len(burst_onsets_one_trial[i]) == 0:
            burst_onsets[i] = -1.0
        elif len(burst_onsets_one_trial[i]) == 1:
            burst_onsets[i] = burst_onsets_one_trial[i][0]
        else:
            print "Neuron {0} has {1} bursts!".format(i, len(burst_onsets_one_trial[i]))
    

    N_TR = 170
    
    # calculate in-degree
    in_degree = np.zeros(N, np.int32)
    for i in range(N):
        for target in targets_ID[i]:
            in_degree[target] += 1


    mean_integration_time_low_degree = []
    
    for i in range(N_TR, N):
        #if (burst_onsets[i] < 150) & (in_degree[i] < 40):
        if in_degree[i] < 40:
            print i, in_degree[i], burst_onsets[i]
            
            inputTimes = get_input_times_neuron(targets_ID, burst_onsets, delays, i)
            mean_integration_time_low_degree.append(np.mean(inputTimes))
            #plt.figure()
            #plt.hist(inputTimes)
            #plt.xlabel('Input time (ms)')
            #plt.ylabel('Count')
            
            #plt.show()

    plt.figure()
    plt.hist(mean_integration_time_low_degree)
    plt.xlabel('Integration time (ms)')
    plt.ylabel('Count')
    
    plt.show()

def plot_input_times(file_connections, file_spikes):
    """
    Plot input times (burst onsets of presynaptic neurons) relative to
    burst onsets of postsynaptic neurons
    """
    ### burst is defined as >= 2 spikes with separation <= 10 ms between spikes 
    INTERSPIKE_INTERVAL = 10.0
    MIN_NUM_SPIKES_IN_BURST = 2
    
    
    burst_onsets_one_trial, num_spikes_in_burst_one_trial = find_burst_onsets(file_spikes, MIN_NUM_SPIKES_IN_BURST, INTERSPIKE_INTERVAL)          
    (N, targets_ID, _, delays) = read_connections(file_connections)
   
    burst_onsets = np.empty(N, np.float32)
    for i in range(N):
        if len(burst_onsets_one_trial[i]) == 0:
            burst_onsets[i] = -1.0
        elif len(burst_onsets_one_trial[i]) == 1:
            burst_onsets[i] = burst_onsets_one_trial[i][0]
        else:
            print "Neuron {0} has {1} bursts!".format(i, len(burst_onsets_one_trial[i]))
    
    inputTimes, delaysLateInputs, fractionLate = get_input_times(targets_ID, burst_onsets, delays)
    
    plt.figure()
    plt.hist(inputTimes, bins=100)
    plt.xlabel('Input time (ms)')
    plt.ylabel('Count')
    
    plt.show()

def get_input_times_neuron(target_ids, burst_times, axonal_delays, neuron_id):
    """
    Function estimates input times relative to the target burst time of the requested neuron
    """
    N = len(target_ids)
    
    inputTimes = []
    
    assert burst_times[neuron_id] > 0
    
    for i in range(N):
        if (neuron_id in target_ids[i]) and (burst_times[i]) > 0:
            for j, target in enumerate(target_ids[i]):
                if (target == neuron_id):
                    time_difference = burst_times[i] + axonal_delays[i][j] - burst_times[target]

                    inputTimes.append(time_difference)
                        
                            
                
    MARGIN_LATE = 0.0

    num_inputs = len(inputTimes)
    num_late_inputs = sum(inp > MARGIN_LATE for inp in inputTimes)

    print "Total number of inputs: ",num_inputs
    print "Number of late inputs: ",num_late_inputs
    print "Fraction of late inputs: ",float(num_late_inputs) / float(num_inputs)
    print "Mean of input times: ",np.mean(inputTimes)
    print "Std of input times: ",np.std(inputTimes)
    
    return inputTimes


def get_input_times(target_ids, burst_times, axonal_delays):
    """
    Function estimates input times relative to the target burst time
    and finds axonal time delays of late inputs
    
    """
    N = len(target_ids)

    inputTimes = []
    delaysLateInputs = [] # axonal time delays of late inputs
    
    for i in range(N):
        if burst_times[i] > 0:
            for j, target in enumerate(target_ids[i]):
                if burst_times[target] > 0:
                    time_difference = burst_times[i] + axonal_delays[i][j] - burst_times[target]
                    if np.fabs(time_difference) >= 30:
                        #print "Difference = {0} {1} -> {2}".format(time_difference, i, target)
                        pass
                    else:
                        inputTimes.append(time_difference)
                        
                        if time_difference > 0:
                            delaysLateInputs.append(axonal_delays[i][j])
                            
                
    MARGIN_LATE = 0.0

    num_inputs = len(inputTimes)
    num_late_inputs = sum(inp > MARGIN_LATE for inp in inputTimes)

    print "Total number of inputs: ",num_inputs
    print "Number of late inputs: ",num_late_inputs
    print "Fraction of late inputs: ",float(num_late_inputs) / float(num_inputs)
    print "Mean of input times: ",np.mean(inputTimes)
    print "Std of input times: ",np.std(inputTimes)
    
    return inputTimes, delaysLateInputs, float(num_late_inputs) / float(num_inputs)


def plot_average_trace(neuron, num_trial, dirname):
    """
    Plot average traces for a neuron
    """
    average_defined = False
    
    average_Vs = None
    average_Vd = None
    average_Gexcd = None
    
    for i in range(num_trial):
        t, Vs, Vd, Gexc_d, _, _, _ = read_hh2(os.path.join(dirname, "testTrial_trial{0}_RA{1}.bin".format(i, neuron)))
        
        if average_defined:    
            average_Vs += Vs
            average_Vd += Vd
            average_Gexcd += Gexc_d
        else:
            average_Vs = Vs
            average_Vd = Vd
            average_Gexcd = Gexc_d
            
            average_defined = True
    
    
# =============================================================================
#     for f in os.listdir(dirname):
#         if "RA" + str(neuron) in f:
#             num_trial += 1
#             t, Vs, Vd, Gexc_d, _, _, _ = read_hh2(os.path.join(dirname, f))
#             
#             if average_defined:    
#                 average_Vs += Vs
#                 average_Vd += Vd
#                 average_Gexcd += Gexc_d
#             else:
#                 average_Vs = Vs
#                 average_Vd = Vd
#                 average_Gexcd = Gexc_d
#                 
#                 average_defined = True
#             
# =============================================================================

    if num_trial > 0:
        average_Vs = average_Vs / float(num_trial)
        average_Vd = average_Vd / float(num_trial)
        average_Gexcd = average_Gexcd / float(num_trial)
        
# =============================================================================
#         plt.figure()
#         plt.plot(t, average_Vs)
#         plt.xlabel('Time (ms)')
#         plt.ylabel('Vs (mV)')
#         
#         plt.figure()
#         plt.plot(t, average_Vd)
#         plt.xlabel('Time (ms)')
#         plt.ylabel('Vd (mV)')
#         
#         plt.figure()
#         plt.plot(t, average_Gexcd)
#         plt.xlabel('Time (ms)')
#         plt.ylabel('Gexc_d (mS/cm^2)')
#         
#         fig, ax1 = plt.subplots()
# 
#         color = 'tab:blue'
#         ax1.set_xlabel('Time (ms)')
#         ax1.set_ylabel('Vs (ms)', color='b')
#         ax1.plot(t, average_Vs, color='b')
#         ax1.tick_params(axis='y', labelcolor='b')
#         
#         ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#         
#         color = 'tab:red'
#         ax2.set_ylabel('Gexc_d (mS/cm^2)', color=color)  # we already handled the x-label with ax1
#         ax2.plot(t, average_Gexcd, color=color)
#         ax2.tick_params(axis='y', labelcolor=color)
# 
# =============================================================================
        f = plt.figure()
        
        ax1 = f.add_subplot(211)
        ax1.plot(t, Vs)
        ax1.plot(t, average_Vs, color='k', linewidth=3.0)
        ax1.set_xlim([0, 1000])
        #ax1.set_ylim([-90, -65])
        ax1.set_ylabel('Vs (ms)')
        
        ax2 = f.add_subplot(212)
        ax2.plot(t, Gexc_d)
        ax2.plot(t, average_Gexcd, color='k', linewidth=3.0)
        ax2.set_xlim([0, 1000])
        #ax2.set_ylim([0, 0.025])
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Gexc_d (mS/cm^2)')
        

        plt.show()        

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

def plot_burst_onset_density(filename, start=-5.0, end=1000.0, bin_width=0.75, min_num_spikes_in_burst=2, max_interspike_interval=10.0):
    """
    Plot density of burst onset times
    """
    burst_onsets, _ = find_burst_onsets(filename, min_num_spikes_in_burst, max_interspike_interval)

    all_burst_onsets = sorted([burst for bursts in burst_onsets for burst in bursts])
    
    time, density = calculate_hist_with_fixed_bin_size(all_burst_onsets, start, end, bin_width)
    
    plot_power_fft(density[(time >= 200) & (time <= 600)], bin_width/1000.0)
    
    plt.figure()
    plt.plot(time, density / bin_width)
    plt.xlabel('Time (ms)')
    plt.ylabel('Burst density (1/ms)')
    
    
    plt.show()
    
def plot_power_fft(signal, dt):
    """
    Plot frequency power of real signal 
    """
    Fk = np.fft.rfft(signal, norm='ortho')
    
    n = len(signal)
    print n
    f = np.fft.fftfreq(n)

    if n % 2 == 0:        
        freq = np.abs(f[0:n/2+1]) / dt
    else:
        freq = f[0:n/2+1] / dt
    
    plt.figure()
    plt.plot(freq, np.absolute(Fk)**2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectral power (a.u.)')
    #ax.set_yscale('log')
    plt.yscale('log')

    plt.show()


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
    (N, spike_times_raw, neuron_fired) = read_spikes(filename)

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

def plot_scaled_weights_results():
    """
    Plots max spectral power in interval (50 Hz, 250 Hz) for networks with scaled excitatory
    conductances
    """
    scaling = [1.0, 2.0, 4.0, 6.0]
    
    spectral_power_integration_times = [3113, 4295, 4381, 4675]
    spectral_power_axonal_delays = [1118, 633, 529, 906]
    
    plt.figure()
    plt.plot(scaling, np.log10(spectral_power_integration_times), '-o', label='integration times')
    plt.plot(scaling, np.log10(spectral_power_axonal_delays), '-o', label='axonal conduction delays')
    plt.xlabel('Excitatory weight scaling')
    plt.ylabel('Log of max spectrial power in (50Hz, 250Hz)')
    plt.legend()
    

def plot_integration_times_grid_results():
    """
    Plots max spectral power of oscillations in burst density in interval (50 Hz, 150 Hz) across 2d grid
    of log-normal integration times distributions
    """
    mean = [5.5, 6.5, 7.5, 8.5]
    std = [0.25, 0.75, 1.25, 1.75, 2.25]
    
    X, Y = np.meshgrid(std, mean)
    
    print X
    print Y
    # freq between 50Hz and 150Hz
    #spectral_power = np.array([[191141, 121163, 141891, 22266, 4363],
    #                           [113578, 81484, 161183, 13767, 1099],
    #                           [131336, 101788, 299852, 25816, None],
    #                           [114095, 114220, 84852, 109604, None]])
    
    # freq between 50Hz and 250Hz
    spectral_power = np.array([[191141, 121163, 141891, 22266, 4363],
                               [130433, 81484, 161183, 13767, 1099],
                               [131336, 101788, 299852, 25816, 964],
                               [114095, 114220, 84852, 109604, 1665]])
    
    
    plt.imshow(np.log10(spectral_power), extent=[0.0,2.5,9,5], aspect='auto')
    plt.title('Log spectral power')
    plt.xlabel('std integration time (ms)')
    plt.ylabel('mean integration time (ms)')
    plt.colorbar()
    plt.xticks(std)
    plt.yticks(mean)
    
    
    print spectral_power
            
def plot_rewiring_results():
    """
    Plots jitter and reliability for simulations with rewired polychronous network
    """
    fraction_rewired = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.65])
    mean_jitter = np.array([0.28, 0.31, 0.38, 0.45, 0.57, 0.88])
    std_jitter = np.array([0.05, 0.05, 0.07, 0.09, 0.15, 0.24])
    fraction_nonreliable = np.array([5e-05, 5e-05, 5e-05, 5e-05, 0.0009, 0.0042])
    
    N = 20000
    
    std_jitter_mean = std_jitter / np.sqrt(float(N) * (1 - fraction_nonreliable))
    
    print std_jitter_mean
    
    plt.figure()
    #plt.plot(fraction_rewired, mean_jitter, '-o')
    plt.errorbar(fraction_rewired, mean_jitter, yerr=std_jitter, fmt='-o')
    plt.xlabel('Fraction of rewired synapses')
    plt.ylabel('Jitter (ms)')
    
    plt.figure()
    plt.plot(fraction_rewired, fraction_nonreliable, '-o')
    plt.xlabel('Fraction of rewired synapses')
    plt.ylabel('Fraction of non-reliable neurons')
    plt.ylim([0, 0.0045])
    plt.locator_params(axis='y', nbins=6)

    
    plt.show()
    
    
           
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
                    
    
    
    # calculate average num bursts in trials when bursts are produced
    average_num_bursts = np.zeros(N, np.float32)
    neuron_reliability = np.zeros(N, np.float32) # defined as a fraction of trials in which bursts are produced 
    
    for i in range(N):
        neuron_reliability[i] = float(len(burst_onsets_trial[i])) / float(num_trials)
        for bursts in burst_onsets_trial[i]:
            average_num_bursts[i] += len(bursts)
        if len(burst_onsets_trial[i]) > 0:
            average_num_bursts[i] = average_num_bursts[i] / float(len(burst_onsets_trial[i]))
    
    plt.figure()
    plt.hist(average_num_bursts)
    plt.xlabel('Average # bursts per trial')
    plt.ylabel('Count')
    
    print "# neurons that produced bursts in less than 50% trials: ",len(np.where(neuron_reliability < 0.5)[0])
    print "Fraction of neurons that produced bursts in less than 50% trials: ",float(len(np.where(neuron_reliability < 0.5)[0]))/float(N)
    
    
    print burst_onsets_trial[0]
    
    
    burst_onsets_allTrials = [[] for _ in range(N)] 
    num_spikes_in_burst_allTrials = [[] for _ in range(N)] 
    
    for i in range(N):
        burst_onsets_allTrials[i] = [item for sublist in burst_onsets_trial[i] for item in sublist]
        num_spikes_in_burst_allTrials[i] = [item for sublist in num_spikes_in_burst_trial[i] for item in sublist]
    
    
    print burst_onsets_allTrials[0]
    
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
    plt.xlim([0, 2.0])
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
    #simDir = "/home/eugene/Programming/data/mlong/integrationConst/grid/grid5_seed1991/testTracesScale4.0/"
    #netDir = "/home/eugene/Programming/data/mlong/integrationConst/grid/grid5_seed1991/"
    
    
    simDir = "/home/eugene/Programming/data/mlong/integrationConst/polyScaled/scaled2.0/"
    netDir = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/"
    
    
    #simDir = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.0/test/"
    #netDir = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.0/"
    
    
    #simDir = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/test/"
    #netDir = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/"
    
    #simDir = "/home/eugene/Programming/data/mlong/integrationConst/poly2/test_same_noise/"
    #netDir = "/home/eugene/Programming/data/mlong/integrationConst/poly2/"
    
    
    #simDir = "/home/eugene/Programming/data/mlong/noise/052519/noise_s0.26_d0.0/"
    #netDir = "/home/eugene/Programming/data/mlong/noise/network/"
    
    #file_jitter = "/home/eugene/Programming/figures/mlong/noise/052519/jitter_relevant_bursts_stdIs0.27.txt"
    #file_burst_density = "/home/eugene/Programming/figures/mlong/noise/052519/average_burst_density_stdIs0.26.txt"
    #file_burst_density = "/home/eugene/Programming/figures/mlong/randomFeedforward/poly/average_burst_density_f0.0.txt"
    #file_jitter = "/home/eugene/Programming/figures/mlong/randomFeedforward/poly/jitter_relevant_bursts_f0.0.txt"
    
    #analyze_spikes_multiple_trials(simDir, netDir, file_jitter)
    #analyze_spikes_multiple_trials(simDir, netDir)
    
    #plot_rewiring_results()
    
    #analyze_dendritic_spikes_multiple_trials(simDir, netDir)
    
    #plot_average_trace(822, 50, simDir) # 822, 1443
    
    
    plot_input_times_low_in_degree("/home/eugene/Programming/data/mlong/integrationConst/gee0.032/poly3/RA_RA_connections.bin", "/home/eugene/Programming/data/mlong/integrationConst/gee0.032/poly3/test/testTrial_0_somaSpikes.bin")
    #plot_input_times("/home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.65/RA_RA_connections.bin", "/home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.65/test/testTrial_0_somaSpikes.bin")
    #plot_input_times("/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/RA_RA_connections.bin", "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/test/testTrial_0_somaSpikes.bin")
    #file_order = "/home/eugene/Programming/data/mlong/noise/052519/noise_s0.0_d0.0/testTrial_0_somaSpikes.bin"
    #file_spikes = "/home/eugene/Programming/data/mlong/noise/052519/noise_s0.27_d0.0/testTrial_0_somaSpikes.bin"
    
    #plot_ordered_noisy_spikes(500.0, file_spikes, file_order)
    #plot_spikes(300.0, "/home/eugene/Programming/data/mlong/noise/052519/noise_s0.0_d0.0/testTrial_0_somaSpikes.bin")
    #plot_spikes(20.0, "/home/eugene/Programming/data/mlong/noise/052519/noise_s0.20_d0.0/testTrial_0_dendSpikes.bin")
    
    #plot_burst_onset_density("/home/eugene/Programming/data/mlong/integrationConst/poly15/e0.004000_i0.000000_somaSpikes.bin")
    
    #plot_integration_times_grid_results()
    #plot_scaled_weights_results()
    
    #plot_burst_onset_density("/home/eugene/Programming/data/mlong/integrationConst/gee0.032/poly2/e0.032000_i0.000000_somaSpikes.bin", end=2000)
    #plot_burst_onset_density("/home/eugene/Programming/data/mlong/integrationConst/grid/grid10_seed7777/test/testTrial_0_somaSpikes.bin", end=2000)
    
    
    #plot_burst_onset_density("/home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.05/test/testTrial_0_somaSpikes.bin")
    #plot_burst_onset_density("/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/test/testTrial_0_somaSpikes.bin")
    
    #plot_spikes(200.0, "/home/eugene/Programming/data/mlong/integrationConst/grid/grid5_seed1991/testTracesScale6.0/testTrial_0_somaSpikes.bin")
    
    #plot_spikes(300.0, "/home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.05/test/testTrial_0_somaSpikes.bin")
    #plot_burst_onset_density("/home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.05/test/testTrial_0_somaSpikes.bin")
    
    #plot_burst_onset_density("/home/eugene/Programming/data/mlong/integrationConst/poly2/test_same_cm/testTrial_0_somaSpikes.bin")
    
    #(N, spike_times_raw, neuron_fired) = read_spikes("/home/eugene/Programming/data/mlong/randomFeedforward/poly/f0.05/test/testTrial_0_somaSpikes.bin")
    #print spike_times_raw[1]
    #print neuron_fired[1]

    #plot_integration_times_grid_results()
    
    #time, average_burst_density = calculate_average_burst_onset_density(simDir, netDir)
    #plt.figure()
    #plt.plot(time, average_burst_density)
    #plt.xlabel('Time (ms)')
    #plt.ylabel('Average burst density (1/ms)')
    #plt.xlim([0, 1000])
    #plt.ylim([0, 170])
    #plt.show()

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
    