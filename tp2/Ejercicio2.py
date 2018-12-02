#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 20:51:31 2018

@author: lisandro
"""

#Librerias

import numpy as np
import matplotlib.pyplot as plt

import pdsmodulos.signal_generator as gen
import pdsmodulos.spectrum_analyzer as sa
import pdsmodulos.statistic as sta
import pdsmodulos.adc as converter

import scipy.signal.windows as win

#Testbench

def testbench():
    
    #Parametros del muestreo
    
    fs = 1024
    N = 1024
    
    #Inicializo el generador de funciones
    
    generador = gen.signal_generator(fs,N)
    
    #Genero el tono x1
    
    fd = 0.5*(fs/N)
    fd = 0
    
    A1 = 1
    f1 = int(np.round(N/4)) + fd
    p1 = 0
    
    (t,x1) = generador.sinewave(A1,f1,p1)

    #Genero el tono x2

    K1 = 40
    K2 = np.power(10,-K1/20)
    
    A2 = A1*K2
    f2 = int(np.round(N/4)) + int(10)
    p2 = 0
    
    (t,x2) = generador.sinewave(A2,f2,p2)

    #Genero el la funcion bitonal

    x = x1 + x2

    plt.figure()
    plt.title('Funcion bitonal')
    plt.plot(t,x)
    plt.xlabel('t[s]')
    plt.ylabel('x(t)[s]')
    plt.grid() 

    #Genera la ventana
    
    w = win.bartlett(N)
    w = np.reshape(w,(N,1))

    #Grafica la ventana
    
    plt.figure()
    plt.plot(t,w)
    plt.title('Ventana')
    plt.ylabel('w(t)[V]')
    plt.xlabel('t[s]')
    plt.grid()

    #Genera la secuencia bitonal ventaneada
    
    x = x*w

    #Grafica la secuencia ventaneada
    
#    plt.figure()
#    plt.plot(t,xw)
#    plt.title('Secuencia ventaneada')
#    plt.ylabel('x(t)[V]')
#    plt.xlabel('t[s]')

    #Enciendo el analizador de espectro
    
    analizador = sa.spectrum_analyzer(fs,N,"fft")
#
#    #Grafico el espectro de la ventana 
#
#    analizador.module_phase(w)

    #Grafico el espectro de la funcion bitonal sin ventanear

    analizador.module_phase(x)
    
#    #Grafico ele espectro de la funcion bitonal ventaneada
#    
#    analizador.module_phase(xw)
    
#Script
    
testbench()