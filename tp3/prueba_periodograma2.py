#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:48:19 2018

@author: lisandro
"""

#Librerias

import numpy as np
import matplotlib.pyplot as plt

import pdsmodulos.signal_generator as gen
import pdsmodulos.spectrum_analyzer as sa
from pandas import DataFrame

#Testbench

def testbench():
    
    #Parametros generales
    S = 50
    N = 1024
    fs = 256
    
    #Parametros del ruido blanco varianza unitaria
    u = 0
    U = u*np.ones((S,1),dtype = float)
    var = 1
    SD = np.sqrt(var)*np.ones((S,1),dtype = float)
    dist = []
    for i in range(0,S):
        dist.append("normal")      
    
    #Enciendo el generador de funciones
    generador = gen.signal_generator(fs,N)
        
    #Genero S realizaciones de del ruido
    (t,n) = generador.noise(dist,U,SD)
  
    #Estimador Periodograma
    (fp,Sxxm_p,Sxxv_p) = sa.periodograma(n,fs,ensemble = True) 
    
    #Grafico todas las realizaciones
    plt.figure()
    plt.plot(fp,20*np.log(Sxxm_p))
    plt.grid()
    
    #Grafico el promedio
    plt.figure()
    plt.plot(fp,20*np.log(np.mean(Sxxm_p,axis = 1)))
    plt.grid()
    
#Script
    
testbench()