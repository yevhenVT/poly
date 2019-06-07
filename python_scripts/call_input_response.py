#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 23:17:41 2019

@author: eugene

Script runs multiple simulations of neuronal responses to input with
difference capacitance of dendritic compartment
"""

import subprocess
import os
import numpy as np
import math


def call_input_response(c, filename):
    """
    Simulate response of a target neuron with given dendritic capacitance to
    a synchronous input of 170 source neurons
    """    
    params = ['/home/eugene/Programming/projects/polychronousPaper/poly/c++/hh2_buffer/responseToInputs/out',\
                    str(c), filename]
        
    subprocess.call(params)


def call_input_response_batch(cm, outDir):
    """
    Simulate responses of a target neuron with different dendritic capacitance to
    a synchronous input of 170 source neurons
    """    
    for c in cm:
        call_input_response(c, os.path.join(outDir, "cm"+str(round(c,1))+".bin"))
        
def call_noise_dend_capacitance_batch(mu_s, std_s, mu_d, std_d, outDir):
    """
    Simulate response of a neuron with different dendritic capacitance to a noise stimulus 
    """

    cm = np.arange(1.0, 5.1, 0.1)
    
    for c in cm:
        call_noise_dend_capacitance(c, mu_s, std_s, mu_d, std_d, os.path.join(outDir, "cm"+str(round(c,1))+".bin"))
        
        
def call_noise_dend_capacitance(c, mu_s, std_s, mu_d, std_d, filename):
    """
    A single simulation of a neuron's response to a noise stimulus with provided dendritic capacitance
    """
    params = ['/home/eugene/Programming/projects/polychronousPaper/poly/c++/hh2_buffer/noiseDiffCm/out',\
                    str(c), str(mu_s), str(std_s), str(mu_d), str(std_d), filename]
        
    subprocess.call(params)


def call_noise_dend_capacitance_batch_model(c, outDir):
    """
    Simulate response of a neuron with different dendritic capacitance to a noise stimulus sampled
    based on dendritic capacitance
    """
    CM_DEND = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    STD_DEND = [0.198, 0.233, 0.263, 0.29, 0.315, 0.34, 0.36]
    
    STD_SOMA_CONST = 0.1
    MU_SOMA_CONST = 0.0
    MU_DEND_CONST = 0.0

    mu_soma = np.empty(len(c), np.float32)
    std_soma = np.empty(len(c), np.float32)
    mu_dend = np.empty(len(c), np.float32)
    std_dend = np.empty(len(c), np.float32)
    
    mu_soma.fill(MU_SOMA_CONST)
    std_soma.fill(STD_SOMA_CONST)
    mu_dend.fill(MU_DEND_CONST)
    
	
    dc = CM_DEND[1] - CM_DEND[0]
    c_min = CM_DEND[0]
	
    for i in range(len(c)):
		ind_floor = int(math.floor((c[i]-c_min) / dc))
		ind_ceil = int(math.ceil((c[i]-c_min) / dc))
		
		alpha = (c[i] - CM_DEND[ind_floor]) / dc
		
		std_dend_neuron = (1-alpha) * STD_DEND[ind_floor] + alpha * STD_DEND[ind_ceil]
							
		std_dend[i] = std_dend_neuron;
        

    for i in range(len(c)):
        print c[i], mu_soma[i], std_soma[i], mu_dend[i], std_dend[i]
        call_noise_dend_capacitance(c[i], mu_soma[i], std_soma[i], mu_dend[i], std_dend[i], os.path.join(outDir, "cm"+str(round(c[i],2))+".bin"))
   


    
    
if __name__ == "__main__":
    outDir = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/integrationTime/sm8.0/"

    cm = np.arange(0.5, 10.1, 0.1)
    print cm
    call_input_response_batch(cm, outDir)
    
    #c = 0.4
    #filename = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/integrationTime/smallCmSmallA/cm0.4_dt0.01.bin"
    #call_input_response(c, filename)
    
    #outDir = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/noise_s0.1_d0.2/"

    #mu_s = 0.0
    #std_s = 0.1
    #mu_d = 0.0
    #std_d = 0.2

    #call_noise_dend_capacitance(mu_s, std_s, mu_d, std_d, outDir)
    
    #filename = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/cm1.0/noise_s0.1_d0.198.bin"

    #c = 1.0
    #mu_s = 0.0
    #std_s = 0.1
    #mu_d = 0.0
    #std_d = 0.198

    #call_noise_dend_capacitance(c, mu_s, std_s, mu_d, std_d, filename)
    
    #outDir = "/home/eugene/Programming/data/mlong/integrationConst/tuneNeuron/same_noise/"
    #c = np.arange(1.0, 4.25, 0.25)
    #print c
    #call_noise_dend_capacitance_batch_model(c, outDir)