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

eps = np.finfo(float).eps

#Testbench

def testbench():
    
    h = np.array([-1,1],dtype = float)
        
    (b,a) = filt.fir_from_impulse_response(h)
    
    w,h = sig.freqz(b,a) 

    plt.figure()
    filt.zplane(b,a)

    plt.figure()
    plt.title('Respuesta de modulo para h = (-1,1)')
    plt.plot(w,20*np.log10(np.abs(h + eps)))
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.title('Respuesta de modulo para h = (-1,1)')
    plt.plot(w,np.angle(h,deg = True))
    plt.grid()
    plt.show()
        
    
#Script
        
testbench()