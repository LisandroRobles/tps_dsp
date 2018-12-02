#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 22:28:54 2018

@author: lisandro
"""

#Librerias

import numpy as np
import pdsmodulos.signal_generator as gen

#Testbench

def testbench():
    
    ###########################
    # Parametros del muestreo #
    ###########################
    
        N = 1000    # Muestras
        fs = 1000   # Hz
    
    ########################################
    # Inicializo el generador de funciones #
    ########################################
    
        generador = gen.signal_generator(fs,N)
    
    ##################
    # a.1) Senoidal #
    #################
    
        a0 = 1      # Volts
        p0 = 0      # Radianes
        f0 = 10     # Decimo bin (forma normalizada)
        
        generador.sinewave(a0,f0,p0)
        
    ##################
    # a.2) Senoidal #
    #################

        a0 = 1                  # Volts
        p0 = 0                  # Radianes
        f0 = np.round(N/2)      # Bin 500 (forma normalizada)
        
        generador.sinewave(a0,f0,p0)
        
    ##################
    # a.3) Senoidal #
    #################
    
        a0 = 1                      # Volts
        p0 = np.pi/2                # Radianes
        f0 = np.round(N/2)          # Bin 500 (forma normalizada)
        
        generador.sinewave(a0,f0,p0)
        
    ##################
    # a.4) Senoidal #
    #################
    
        a0 = 1                      # Volts
        p0 = 0                      # Radianes
        f0 = np.round(N) + 10       # Bin 500 (forma normalizada)
        
        generador.sinewave(a0,f0,p0)
        
#Script

testbench()
        
        
    
    