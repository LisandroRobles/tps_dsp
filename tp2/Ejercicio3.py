#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:44:33 2018

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
    N = np.power(2,10)
    fs = np.power(2,10)
    
    #Size del montecarlo
    S = 200
    
    #Limites de la distribucion de fr
    l1 = -2*(fs/N)
    l2 = 2*(fs/N)
    
    #Amplitud
    Ao = 2*np.ones((S,1),dtype = float)
    
    #Fase
    Po = np.zeros((S,1),dtype = float)
    
    #Frecuencia central
    fo = int(np.round(N/4))
    
    #Cantidad ventanas
    Nw = 2
    
    #Fr sera una variable aleatoria de distribucion uniforme entre -2 y 2
    #Genero 200 realizaciones de fr
    fr = np.random.uniform(l1,l2,S).reshape(S,1)

    #Genero 200 realizaciones de f1
    f1 = fo + fr

    #Enciendo el generador de funciones
    generador = gen.signal_generator(fs,N)
    
    #Genero 200 realizaciones de x
    (t,x) = generador.sinewave(Ao,f1,Po)

    #Genera una matriz con las 5 ventanas
    w = np.array([],dtype='float').reshape(N,0)

    for j in range(0,Nw):
        
        if j is 0:
            wj = np.ones((N,1),dtype = float)
        elif j is 1:
            wj = win.flattop(N).reshape(N,1)
        elif j is 2:
            wj = win.hann(N).reshape(N,1)
        elif j is 3:
            wj = win.blackman(N).reshape(N,1)
        elif j is 4:
            wj = win.flattop(N).reshape(N,1)
    
        w = np.hstack([w, wj.reshape(N,1)])

    
    #Inicializo el analizador de espectro
    analizador = sa.spectrum_analyzer(fs,N,algorithm = "fft")
    
    for j in range(0,Nw):
    
        wi = w[:,j].reshape(N,1)
        
        Ao_estimador = np.array([],dtype='float').reshape(1,0)
        
        #Contempla la energia o potencia en un ancho de banda
        Ao2_estimador = np.array([],dtype='float').reshape(1,0) 
        
        for i in range(0,S):
            
            xi = x[:,i].reshape(N,1)
            
            xi = xi*wi
            
            #Obtengo el espectro para las i esima realizacion de x
            (f,Xi) = analizador.module(xi)
          
            aux = Xi[(fo - 2):(fo + 3)]
            
            #Calculo una realizacion del estimador
            Aoi_estimador = Xi[fo].reshape(1,1)
            
            #Ao2i_estimador = np.sqrt(np.sum(np.power(aux,2))).reshape(1,1)
            Ao2i_estimador = np.sqrt(np.sum(np.power(aux,2))/5).reshape(1,1)
            
            #Lo adjunto a los resultados
            Ao_estimador = np.hstack([Ao_estimador, Aoi_estimador])
            Ao2_estimador = np.hstack([Ao2_estimador, Ao2i_estimador])
    
        plt.figure()
        plt.hist(np.transpose(Ao_estimador),bins = 'auto')
        plt.axis('tight')
        plt.title('Energia en el bin para ventana #' + str(j))
        plt.xlabel('Variable aleatoria')
        plt.ylabel('Probabilidad')
        plt.grid()    
    
        plt.figure()
        plt.hist(np.transpose(Ao2_estimador),bins = 'auto')
        plt.title('Energia en un ancho de banda para ventana #' + str(j))
        plt.axis('tight')
        plt.xlabel('Variable aleatoria')
        plt.ylabel('Probabilidad')
        plt.grid()    
    
        sesgo = np.mean(Ao_estimador) - Ao[0,0]
        
        varianza = np.mean(np.power(Ao_estimador-np.mean(Ao_estimador),2))
    
        sesgo2 = np.mean(Ao2_estimador) - Ao[0,0]
        
        varianza2 = np.mean(np.power(Ao2_estimador-np.mean(Ao2_estimador),2))
    
        print('------------Ventana #' + str(j) + '--------------')
    
        print('Estimador: Energia en un bin')
    
        print('Sesgo: ' + str(sesgo))
        
        print('Varianza: ' + str(varianza))
    
        print('Estimador: Energia en un ancho de banda (5 bin)')
    
        print('Sesgo: ' + str(sesgo2))
        
        print('Varianza: ' + str(varianza2))

        i = i + 1

    return w

#Script
    
w = testbench()