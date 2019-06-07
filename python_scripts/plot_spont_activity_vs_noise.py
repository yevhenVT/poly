#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:53:19 2019

@author: eugene
"""
import matplotlib.pyplot as plt

#spont_activity = [0.0, 0.0, 0.0, 0.0, 0.02, 0.1, 0.6]
#std_v = [4.3, 4.6, 5.0, 5.5, 5.9, 6.5, 7.1]

#sd = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

std_Is = [0.1, 0.12, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]
std_Vs = [3.48, 4.18, 5.23, 5.58, 5.94, 6.30, 6.67, 7.04, 7.41, 7.81, 8.20, 8.60, 9.00, 9.38, 9.79, 10.20, 10.59, 11.01]
spont_activity = [0.0, 0.0, 0.0, 0.0, 0.01, 0.03, 0.05, 0.08, 0.14, 0.33, 0.51, 0.73, 1.13, 1.49, 1.93, 2.54, 3.07, 3.79]

plt.figure()
plt.plot(std_Vs, spont_activity, '-o')
plt.xlabel('Std V (mV)')
plt.ylabel('Spontaneous activity (Hz)')

plt.figure()
plt.plot(std_Is, spont_activity, '-o')
plt.xlabel('Std Is (nA)')
plt.ylabel('Spontaneous activity (Hz)')


plt.show()
