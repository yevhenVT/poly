#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 18:16:35 2019

@author: eugene

Script generates file with connections to use in a newer version of simulations (on polychronousPaper)
from files used in an older version (on nospace)
"""
import struct

SIZE_OF_INT = 4
SIZE_OF_DOUBLE = 8

def read_connections_old(filename):
    """
    Read neuronal output connections generated from experimental data from binary file
    """
    with open(filename, 'rb') as file:
        data = file.read()
        #print len(data)
        
        N = struct.unpack("<i", data[:SIZE_OF_INT])[0]
        
        print N
        #print N_RA     
        
        targets_ID = [] # ids of target neurons
        syn_length = [] # synaptic lengths
        syn_G = [] # synaptic conductances
        
        #targets_RA_I_ID = []
        #targets_RA_I_G = []    
        ind = SIZE_OF_INT    
        
        for i in range(N):
            neurons_ID = struct.unpack("<i", data[ind:(ind+SIZE_OF_INT)])[0]
            number_of_targets = struct.unpack("<i", data[(ind+SIZE_OF_INT):(ind+2*SIZE_OF_INT)])[0]
            
            temp_ID = []
            temp_length = []
            temp_G = []    
            
            ind += 2*SIZE_OF_INT        
            
            for j in range(number_of_targets):
                
                ID = struct.unpack("<i", data[ind:(ind + SIZE_OF_INT)])[0]
                L = struct.unpack("<d", data[((ind + SIZE_OF_INT)):(ind + SIZE_OF_INT + SIZE_OF_DOUBLE)])[0]
                G = struct.unpack("<d", data[((ind + SIZE_OF_INT + SIZE_OF_DOUBLE)):(ind + SIZE_OF_INT + 2*SIZE_OF_DOUBLE)])[0]
                
                
                temp_ID.append(ID)
                temp_length.append(L)
                temp_G.append(G)
                
                ind += SIZE_OF_INT + 2*SIZE_OF_DOUBLE
            
            targets_ID.append(temp_ID)
            syn_length.append(temp_length)        
            syn_G.append(temp_G)
            
        return (N, targets_ID, syn_length, syn_G)

def read_axonal_delays_old(filename):
    """
    Read axonal time delays from file
    """
    with open(filename, "rb") as file:
        data = file.read()
        
        N = struct.unpack("<i", data[0:SIZE_OF_INT])[0]
        
        axonal_delays = []
        syn_lengths = []
        
        start_ind = SIZE_OF_INT  
        
        for i in range(N):   
            # get number of targets
            num_targets =  struct.unpack("<i", data[start_ind:(start_ind+SIZE_OF_INT)])[0]
            
            start_ind += SIZE_OF_INT
            
            axonal_delays_tmp = []
            syn_lengths_tmp = []
            
            for j in range(num_targets):
                axonal_delays_tmp.append(struct.unpack("<d", data[start_ind:(start_ind+SIZE_OF_DOUBLE)])[0])
                syn_lengths_tmp.append(struct.unpack("<d", data[(start_ind+SIZE_OF_DOUBLE):(start_ind+2*SIZE_OF_DOUBLE)])[0])
                                        
                start_ind += 2*SIZE_OF_DOUBLE
              
            axonal_delays.append(axonal_delays_tmp)
            syn_lengths.append(syn_lengths_tmp)
            
        
       
        return axonal_delays, syn_lengths

def make_new_connection_file(file_connections_old, file_delays_old, file_connections_new):
    """
    Script takes old files with connections and delays and create a new file with connections
    """
    (N, targets_ID, _, syn_G) = read_connections_old(file_connections_old)
    delays, _ = read_axonal_delays_old(file_delays_old)
    
    with open(file_connections_new, "wb") as f:
        f.write(struct.pack("<i", N))
        
        for i in range(N):
            f.write(struct.pack("<i", i))
            f.write(struct.pack("<i", len(targets_ID[i])))
            
            for j in range(len(targets_ID[i])):
                f.write(struct.pack("<i", targets_ID[i][j]))
                f.write(struct.pack("<d", syn_G[i][j]))
                f.write(struct.pack("<d", delays[i][j]))
                
    
if __name__ == "__main__":
    file_connections_RA2RA_old = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/old/RA_RA_connections.bin"
    file_delays_RA2RA_old = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/old/delays_RA2RA.bin"
    file_connections_RA2RA_new = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/RA_RA_connections.bin"
    
    
    file_connections_RA2I_old = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/old/RA_I_connections.bin"
    file_delays_RA2I_old = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/old/delays_RA2I.bin"
    file_connections_RA2I_new = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/RA_I_connections.bin"
    
    file_connections_I2RA_old = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/old/I_RA_connections.bin"
    file_delays_I2RA_old = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/old/delays_I2RA.bin"
    file_connections_I2RA_new = "/home/eugene/Programming/data/mlong/randomFeedforward/poly/network/new/I_RA_connections.bin"
    
    #make_new_connection_file(file_connections_RA2RA_old, file_delays_RA2RA_old, file_connections_RA2RA_new)
    make_new_connection_file(file_connections_RA2I_old, file_delays_RA2I_old, file_connections_RA2I_new)
    make_new_connection_file(file_connections_I2RA_old, file_delays_I2RA_old, file_connections_I2RA_new)
    
    #from show_axonal_conduction_dist import read_connections
    #(N, targets_ID, weights, delays) = read_connections(file_connections_RA2RA_new)
    
    
    #(N_old, targets_ID_old, syn_length_old, weights_old) =  read_connections_old(file_connections_RA2RA_old)
    #axonal_delays_old, syn_lengths = read_axonal_delays_old(file_delays_RA2RA_old)
    
    #assert N == N_old
    #assert targets_ID == targets_ID_old
    #assert weights == weights_old
    #assert axonal_delays_old == delays
    #assert syn_lengths == syn_length_old
    
    #print len(targets_ID)
    #print len(weights)
    #print len(delays)
    
    #print targets_ID[0]
    #print weights[0]
    #print delays[0]

    #(N, targets_ID, syn_length, syn_G) = read_connections_old(file_connections_RA2RA_old)
    
# =============================================================================
#     print len(targets_ID)
#     print len(syn_length)
#     print len(syn_G)
#     
#     print targets_ID[0]
#     print syn_length[0]
#     print syn_G[0]
#     
#     
#     axonal_delays, syn_lengths = read_axonal_delays_old(file_delays_RA2RA_old)
#     
#     print len(axonal_delays)
#     print len(syn_lengths)
#     
#     print axonal_delays[0]
#     print syn_lengths[0]
# =============================================================================
