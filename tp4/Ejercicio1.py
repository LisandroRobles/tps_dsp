#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:43:05 2018

@author: lisandro
"""


#Paquetes

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import pdsmodulos.filters as filt
    
#Testbench

def testbench():
        
    N = np.array([3,4,5],dtype = int)
    
    for Ni in N:
        
        bi,ai = filt.moving_average(Ni)
                
        wi,hi = sig.freqz(bi,ai) 

        plt.figure()
        plt.title('Diagrama de polos y ceros para N = ' + str(Ni))
        filt.zplane(bi,ai)

        plt.figure()
        plt.title('Respuesta de modulo para N = ' + str(Ni))
        plt.plot(wi,20*np.log10(np.abs(hi)))
        plt.grid()
        
        plt.figure()
        plt.title('Respuesta de fase para N = ' + str(Ni))
        plt.plot(wi,np.angle(hi,deg = True))
        plt.grid()
    
#Script
        
testbench()