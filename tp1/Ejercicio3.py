#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 22:54:16 2018

@author: lisandro
"""

#Librerias

import numpy as np
import pandas as pd
from IPython.display import HTML

import pdsmodulos.signal_generator as gen
import pdsmodulos.spectrum_analyzer as sa
import pdsmodulos.statistic as sta

#Testbench

def testbench():
    
    ###########################
    # Parametros del muestreo #
    ###########################
    
        N = 1024    # Muestras
        fs = 1024   # Hz
        
    ########################################
    # Inicializo el generador de funciones #
    ########################################
    
        generador = gen.signal_generator(fs,N)
        
    ########################################
    # Inicializo el analizador de espectro #
    ########################################
    
        analizador = sa.spectrum_analyzer(fs,N,algorithm = "fft")

    ########################################
    # Genera secuencia patron              #
    ########################################
    
        a0 = 1                  # Volts
        p0 = 0                  # Radianes
        fn = int(np.round(N/4)) # Bin N/4 (forma normalizada)
        
    ##################
    # a) Leakege     #
    ##################
    
        fd = (0,0.01,0.25,0.5)  # Frecuencias de desintonia  
        
        tus_resultados = [ ['$ \lvert X(f_0) \lvert$', '$ \lvert X(f_0+1) \lvert $', '$\sum_{i=F} \lvert X(f_i) \lvert$'], 
                   ['',                        '',                           '$F:f \neq f_0$']]
        
        for fd_actual in fd:
            
            resto_frecuencias = 0
            
            f0 = fn + fd_actual
            
            (t,x) = generador.sinewave(a0,f0,p0)
            (f,X_mod,X_ph) = analizador.module_phase(x)
            
            frec_central = 2*X_mod[fn] if (fn != 0 and fn != N/2) else X_mod[fn] 
            primer_adyacente = 2*X_mod[fn+1] if ((fn + 1) != 0 and (fn + 1) != N/2) else X_mod[fn + 1]
            resto_frecuencias = 0
            
            for i in range(0,np.size(X_mod,0)):
                if i != fn and i != (fn + N/2):
                    Xi = X_mod[i]
                    resto_frecuencias = resto_frecuencias + Xi
                
            tus_resultados.append([str(frec_central),str(primer_adyacente),str(resto_frecuencias)])
            
        df = pd.DataFrame(tus_resultados)
        
    #######################
    # b) Zero padding     #
    #######################

        Mj = (int(np.round(10*N)),)
        
        (t,x) = generador.sinewave(a0,fn,p0)
    
        analizador.module_phase(x)
        
        for Mi in Mj:
                        
            (t,xi) = generador.zero_padding(x,Mi)
            Ni = generador.N + Mi            
            analizador.set_points(Ni)
            analizador.module_phase(xi)
        
#Script
            
df = testbench()
        
        

        