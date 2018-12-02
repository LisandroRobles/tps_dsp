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
        
    N = np.array([2,3,4],dtype = int)
    b = -1
    
    for Ni in N:
        
        bi,ai = filt.input_delay(Ni,b)
                
        wi,hi = sig.freqz(bi,ai) 

        plt.figure()
        filt.zplane(bi,ai)

        plt.figure()
        plt.title('Respuesta de modulo para entrada demorada N = ' + str(Ni) + ' muestras')
        plt.plot(wi,20*np.log10(np.abs(hi)))
        plt.grid()
        plt.show()
        
        plt.figure()
        plt.title('Respuesta de fase para entrada demorada N = ' + str(Ni) + ' muestras')
        plt.plot(wi,np.angle(hi,deg = True))
        plt.grid()
        plt.show()
        
    
#Script
        
testbench()