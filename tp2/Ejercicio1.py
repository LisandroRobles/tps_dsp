#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 20:51:31 2018

@author: lisandro
"""

#Librerias

import numpy as np
import matplotlib.pyplot as plt

import pdsmodulos.spectrum_analyzer as sa
import pdsmodulos.windows as win

#Testbench

def testbench():
    
    #Parametros del muestreo    
    N1 = np.power(2,10)     #Cantidad de puntos de la ventana
    N2 = np.power(2,20)   #Cantidad de puntos de la fft (zero pad para ver el espectro)
    fs = np.power(2,10)   #Frecuencia de muestreo

    #Inicializo el analizador de espectro
    SA = sa.spectrum_analyzer(fs,N2)

    #Generacion de una ventana rectangular  
    wr = win.rectangular(N1)

#    #Presentacion grafica de los resultados temporales
#    plt.figure()
#    plt.title('Ventana Rectangular, N = ' + str(N1))
#    plt.plot(wr)
#    plt.xlabel('n')
#    plt.ylabel('wr[n]')
#    plt.grid()
        
    #Calculo del espectro de la ventana rectangular (kernel de Dirichlet)
    (f,Wr) = SA.module(wr,db = True)
    
    plt.figure()
    plt.title('Espectro de modulo de ventana rectangular (kernel de Dirichlet), N = ' + str(N1))
    plt.plot(f[0:int(N2/32)],Wr[0:int(N2/32),0])
    plt.xlabel('f[Hz]')
    plt.ylabel('Wr[V/Hz]')
    plt.grid()
    
    #Generacion de una ventana bartlett(triangular)    
    wt = win.bartlett(N1)

#    #Presentacion grafica de los resultados temporales
#    plt.figure()
#    plt.title('Ventana de Bartlett, N = ' + str(N1))
#    plt.plot(wt)
#    plt.xlabel('n')
#    plt.ylabel('wt[n]')
#    plt.grid()
        
    #Calculo del espectro de la ventana barlett
    (f,Wt) = SA.module(wt,db = True)
    
    plt.figure()
    plt.title('Espectro de modulo de ventana bartlett, N = ' + str(N1))
    plt.plot(f[0:int(N2/32)],Wt[0:int(N2/32),0])
    plt.xlabel('f[Hz]')
    plt.ylabel('Wt[V/Hz]')
    plt.grid()

    #Generacion de una ventana hann   
    whn = win.hann(N1)

#    #Presentacion grafica de los resultados temporales
#    plt.figure()
#    plt.title('Ventana de Hann, N = ' + str(N1))
#    plt.plot(whn)
#    plt.xlabel('n')
#    plt.ylabel('whn[n]')
#    plt.grid()
        
    #Calculo del espectro de la ventana de hann
    (f,Whn) = SA.module(whn,db = True)
    
    plt.figure()
    plt.title('Espectro de modulo de ventana de hann, N = ' + str(N1))
    plt.plot(f[0:int(N2/32)],Whn[0:int(N2/32),0])
    plt.xlabel('f[Hz]')
    plt.ylabel('Whn[V/Hz]')
    plt.grid()        

    #Generacion de una ventana blackman    
    wb = win.blackman(N1)

#    #Presentacion grafica de los resultados temporales
#    plt.figure()
#    plt.title('Ventana de Blackman, N = ' + str(N1))
#    plt.plot(whn)
#    plt.xlabel('n')
#    plt.ylabel('wb[n]')
#    plt.grid()
        
    #Calculo del espectro de la ventana de hann
    (f,Wb) = SA.module(wb,db = True)
    
    plt.figure()
    plt.title('Espectro de modulo de ventana de blackman, N = ' + str(N1))
    plt.plot(f[0:int(N2/32)],Wb[0:int(N2/32),0])
    plt.xlabel('f[Hz]')
    plt.ylabel('Wb[V/Hz]')
    plt.grid()     

    #Generacion de una ventana flat-top    
    wft = win.flattop(N1)

#    #Presentacion grafica de los resultados temporales
#    plt.figure()
#    plt.title('Ventana Flat-Top, N = ' + str(N1))
#    plt.plot(wft)
#    plt.xlabel('n')
#    plt.ylabel('wft[n]')
#    plt.grid()
        
    #Calculo del espectro de la ventana float-top
    (f,Wft) = SA.module(wft,db = True)
    
    plt.figure()
    plt.title('Espectro de modulo de ventana flat-top, N = ' + str(N1))
    plt.plot(f[0:int(N2/32)],Wft[0:int(N2/32),0])
    plt.xlabel('f[Hz]')
    plt.ylabel('Wft[V/Hz]')
    plt.grid() 
    
#Script
testbench()