#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:17:10 2017

@author: jingroup

script analyzes tuning of HVC-RA neuron
"""
import reading
import matplotlib.pyplot as plt
import numpy as np
import os

G_L = 0.1

def calculate_spike_properties(t, V, spike_threshold):
    """
    Calculate spike_height, spike width and inter-spike interval for voltage-time data
    
    spike_height is difference between peak of membrane potential in the first spike and the lowest potential after
    first spike
    
    spike width is width of the first spike at half spike amplitude
    
    inter-spike interval is distance between peaks of first two spikes
    """
    # find locations of spikes
    t_threshold_crossing = t[np.where(np.diff(V >= spike_threshold))]

    #print len(t_threshold_crossing)
    #print "threshold_crossing:",t_threshold_crossing

    if len(t_threshold_crossing) > 2:
        Vmax_first = np.max(V[(t > t_threshold_crossing[0]) & (t < t_threshold_crossing[1])])
        Vmin_first = np.min(V[(t > t_threshold_crossing[1]) & (t < t_threshold_crossing[1] + 5)])
        
        tind = np.argmax(V[(t > t_threshold_crossing[0]) & (t < t_threshold_crossing[1])])
        
        tmax_first = t[(t > t_threshold_crossing[0]) & (t < t_threshold_crossing[1])][tind]
        
        #print "Vmax, Vmin = ",Vmax_first, Vmin_first
        
        spike_amplitude = Vmax_first - Vmin_first
        half_spike_height = (Vmax_first + Vmin_first) / 2.
        
        # find crossing at half spike height
        t_hs_crossing = t[np.where(np.diff(V >= half_spike_height))]
        #print "hs_crossing:",t_hs_crossing
   
        spike_width = t_hs_crossing[1] - t_hs_crossing[0]
    else:
        spike_width = -1
        spike_amplitude = -1
        
    if len(t_threshold_crossing) > 3:
        # find peak of second spike
        
        tind = np.argmax(V[(t > t_threshold_crossing[2]) & (t < t_threshold_crossing[3])])
        
        tmax_second = t[(t > t_threshold_crossing[2]) & (t < t_threshold_crossing[3])][tind]
        
        interspike_interval = tmax_second - tmax_first
    else:
        interspike_interval = -1
        
    return spike_amplitude, spike_width, interspike_interval

def get_spike_properties_in_response_to_conductance_peak(dirname, spike_threshold):
    """
    Calculate spike width and inter-spike intervals for all files with responses to
    excitatory conductance kick
    """
    files = os.listdir(dirname)
    
    Gkick = []
    spike_amplitudes = []
    spike_widths = []
    interspike_intervals = []
    
    for f in files:
        if ("RA" in f) and (not "_dend_" in f) and (not "_soma_" in f):
            print f
            t, Vs, Vd, Gexc_d, Ginh_d, n,\
                 h, r, c, Ca = reading.read_hh2_buffer_full(os.path.join(dirname, f))
    
            spike_amplitude, spike_width, interspike_interval = calculate_spike_properties(t, Vs, spike_threshold)
            
            Gkick.append(np.max(Gexc_d) / G_L)
            spike_amplitudes.append(spike_amplitude)
            spike_widths.append(spike_width)
            interspike_intervals.append(interspike_interval)
    
    Gkick, spike_amplitudes, spike_widths, interspike_intervals = zip(*sorted(zip(Gkick, spike_amplitudes, spike_widths, interspike_intervals)))
    
    return Gkick, spike_amplitudes, spike_widths, interspike_intervals

dirname_original = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/cm1.0/"
dirname_tuned = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/cm5.0/"


file_original_soma = os.path.join(dirname_original, "fI_soma.bin")
file_original_dend = os.path.join(dirname_original, "fI_dend.bin")

file_tuned_soma = os.path.join(dirname_tuned, "fI_soma.bin")
file_tuned_dend = os.path.join(dirname_tuned, "fI_dend.bin")


file_original_conductance_response = os.path.join(dirname_original, "G_response.bin")
file_tuned_conductance_response = os.path.join(dirname_tuned, "G_response.bin")

ampl_soma_original, num_spikes_soma_original, _, = reading.read_fI_HVCRA(file_original_soma)
ampl_dend_original, num_spikes_dend_original, _, = reading.read_fI_HVCRA(file_original_dend)

ampl_soma_tuned, num_spikes_soma_tuned, _, = reading.read_fI_HVCRA(file_tuned_soma)
ampl_dend_tuned, num_spikes_dend_tuned, _, = reading.read_fI_HVCRA(file_tuned_dend)

G_original, burst_onset_times_original, spike_times_original = reading.read_conductance_response(file_original_conductance_response)
G_tuned, burst_onset_times_tuned, spike_times_tuned = reading.read_conductance_response(file_tuned_conductance_response)



plt.figure()

label_original = 'cm = 1.0'
label_tuned = 'cm = 0.5'

plt.plot(ampl_soma_original, num_spikes_soma_original, '-bo', label=label_original)
plt.plot(ampl_soma_tuned, num_spikes_soma_tuned, '-ro', label=label_tuned)

plt.xlabel('I (nA)')
plt.ylabel('# of spikes')
plt.title('Injection to soma')
plt.legend()

plt.figure()

plt.plot(ampl_dend_original, num_spikes_dend_original, '-bo', label=label_original)
plt.plot(ampl_dend_tuned, num_spikes_dend_tuned, '-ro', label=label_tuned)

plt.xlabel('I (nA)')
plt.ylabel('# of spikes')
plt.title('Injection to dend')
plt.legend()

plt.figure()

plt.plot(G_original / G_L, spike_times_original, '-bo', label=label_original)
plt.plot(G_tuned / G_L, spike_times_tuned, '-ro', label=label_tuned)

plt.xlabel('G_kick (G_L)')
plt.ylabel('Time to first spike (ms)')

plt.legend()


plt.figure()

plt.plot(G_original / G_L, burst_onset_times_original, '-bo', label=label_original)
plt.plot(G_tuned / G_L, burst_onset_times_tuned, '-ro', label=label_tuned)

plt.xlabel('G_kick (G_L)')
plt.ylabel('Time to burst onset (ms)')

plt.legend()


# calculate spike width and interspike intervals


spike_threshold = 0.0

Gkick_original, spike_amplitude_original, spike_width_original, interspike_interval_original = \
                get_spike_properties_in_response_to_conductance_peak(dirname_original, spike_threshold)
                
Gkick_tuned, spike_amplitude_tuned, spike_width_tuned, interspike_interval_tuned = \
                get_spike_properties_in_response_to_conductance_peak(dirname_tuned, spike_threshold)

# compare neuron traces
file_original_trace = os.path.join(dirname_original, "RA8.bin")
file_tuned_trace = os.path.join(dirname_tuned, "RA8.bin")

t_original, Vs_original, Vd_original, Gexc_d_original, Ginh_d_original, n_original,\
             h_original, r_original, c_original, Ca_original = reading.read_hh2_buffer_full(file_original_trace)

t_tuned, Vs_tuned, Vd_tuned, Gexc_d_tuned, Ginh_d_tuned, n_tuned,\
             h_tuned, r_tuned, c_tuned, Ca_tuned = reading.read_hh2_buffer_full(file_tuned_trace)

#print calculate_spike_properties(t_original, Vs_original, -20.0)
#print calculate_spike_properties(t_tuned, Vs_tuned, -20.0)

f = plt.figure()

ax1 = f.add_subplot(311)
ax1.plot(Gkick_original, spike_amplitude_original, 'b', label=label_original)
ax1.plot(Gkick_tuned, spike_amplitude_tuned, 'r', label=label_tuned)
ax1.set_ylabel("spike amplitude (mV)")
ax1.legend()

ax2 = f.add_subplot(312)
ax2.plot(Gkick_original, spike_width_original, 'b', label=label_original)
ax2.plot(Gkick_tuned, spike_width_tuned, 'r', label=label_tuned)
ax2.set_ylabel("spike width (ms)")
ax2.legend()

ax3 = f.add_subplot(313)
ax3.plot(Gkick_original, interspike_interval_original, 'b', label=label_original)
ax3.plot(Gkick_tuned, interspike_interval_tuned, 'r', label=label_tuned)
ax3.set_ylabel("ISI (ms)")
ax3.set_xlabel("Gkick ($G_L$)")
ax3.legend()


# membrane potentials
f = plt.figure()

ax1 = f.add_subplot(211)
ax1.plot(t_original, Vs_original, 'b', label=label_original)
ax1.plot(t_tuned, Vs_tuned, 'r', label=label_tuned)
ax1.set_ylabel("Vs (mV)")
ax1.legend()

ax2 = f.add_subplot(212)
ax2.plot(t_original, Vd_original, 'b', label=label_original)
ax2.plot(t_tuned, Vd_tuned, 'r', label=label_tuned)
ax2.set_ylabel("Vd (mV)")
ax2.set_xlabel("Time (ms)")
ax2.legend()

# conductances
f = plt.figure()

ax1 = f.add_subplot(211)
ax1.plot(t_original, Gexc_d_original, 'b', label=label_original)
ax1.plot(t_tuned, Gexc_d_tuned, 'r', label=label_tuned)
ax1.set_ylabel("Gexc_d (mS/cm^2)")
ax1.legend()

ax2 = f.add_subplot(212)
ax2.plot(t_original, Ginh_d_original, 'b', label=label_original)
ax2.plot(t_tuned, Ginh_d_tuned, 'r', label=label_tuned)
ax2.set_ylabel("Ginh_d (mS/cm^2)")
ax2.set_xlabel("Time (ms)")
ax2.legend()

# dendritic compartment variables
f = plt.figure()

ax1 = f.add_subplot(311)
ax1.plot(t_original, r_original, 'b', label=label_original)
ax1.plot(t_tuned, r_tuned, 'r', label=label_tuned)
ax1.set_ylabel("r")
ax1.legend()

ax2 = f.add_subplot(312)
ax2.plot(t_original, Ca_original, 'b', label=label_original)
ax2.plot(t_tuned, Ca_tuned, 'r', label=label_tuned)
ax2.set_ylabel("Ca")
ax2.legend()

ax3 = f.add_subplot(313)
ax3.plot(t_original, c_original, 'b', label=label_original)
ax3.plot(t_tuned, c_tuned, 'r', label=label_tuned)
ax3.set_ylabel("c")
ax3.legend()
ax3.set_xlabel("Time (ms)")

# somatic compartment variables
f = plt.figure()

ax1 = f.add_subplot(211)
ax1.plot(t_original, n_original, 'b', label=label_original)
ax1.plot(t_tuned, n_tuned, 'r', label=label_tuned)
ax1.set_ylabel("n")
ax1.legend()

ax2 = f.add_subplot(212)
ax2.plot(t_original, h_original, 'b', label=label_original)
ax2.plot(t_tuned, h_tuned, 'r', label=label_tuned)
ax2.set_ylabel("h")
ax2.set_xlabel("Time (ms)")
ax2.legend()


plt.show()