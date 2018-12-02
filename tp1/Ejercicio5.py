#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 00:12:22 2018

@author: lisandro
"""

#Librerias

import numpy as np
import matplotlib.pyplot as plt

import pdsmodulos.signal_generator as gen
import pdsmodulos.spectrum_analyzer as sa
import pdsmodulos.statistic as sta
import pdsmodulos.adc as converter

#Testbench

def testbench():
    
    #############################
    # Parametros del generador  #
    #############################
    
        #Se busca una grilla temporal densa que simule un caso continuo
        N_t = np.power(2,20)       # Muestras
        fs_t = np.power(2,20)      # Hz
    
    ########################################
    # Inicializo el generador de funciones #
    ########################################
    
        generador = gen.signal_generator(fs_t,N_t)

    ##########################################
    # Generacion de la señal deterministica  #
    ##########################################    
    
        A0 = np.sqrt(2)
        f0 = 1
        p0 = 0
    
        (t,s) = generador.sinewave(A0,f0,p0)

        #Potencia de la señal
        Ps = np.var(s)

        #Energia normalizada de la señal(en un periodo)
        Es = (Ps)/f0
    
        #Varianza de la senal
        print('La varianza de la señal perfecta es: ' + str(np.var(s)))
    
    ##########################################
    # Generacion de la señal aleatoria       #
    ##########################################    
        
        k = Es/200
    
        dist = ("uniform",)
        a1 = np.sqrt(k)*np.sqrt(3)
        #a1 = 0
        
        #Potencia del ruido
        a2 = -a1
        #a2 = 0
    
        (t,n) = generador.noise(dist,a1,a2)

        #Varianza de la senal
        print('La varianza del ruido analogico es: ' + str(np.var(n)))
        
        
    ##########################################
    # Generacion de la señal real            #
    ##########################################    
    
        sn = s + n

        print('La varianza de la señal contaminada es: ' + str(np.var(sn)))

        
        #Presentacion temporal de los resultados
        plt.figure()
        plt.title('Señal con ruido')
        plt.plot(t,sn)#,plt.legend(label = ['SNR = ' + str(SNR) + ' dB'],loc = 'upper right')
        plt.axis('tight')
        plt.xlabel('t[s]')
        plt.ylabel('sn(t)[V]')
        plt.grid()
        
    #############################
    # Parametros del ADC        #
    #############################
    
        n = 16
        Vref_high = np.sqrt(2)
        Vref_low = -np.sqrt(2)
        fs = np.power(2,10)
    
    ########################################
    # Inicializo el ADC                    #
    ########################################
    
        ADC = converter.adc(fs,Vref_low,Vref_high,n)

    ########################################
    # Simulo el muestreo con el ADC        #
    ########################################
        
        (t,sn) = ADC.muestrear(t,sn)

        print('La varianza de la señal contaminada muestreada es: ' + str(np.var(sn)))
        
        N = np.size(t,0)
        
    ########################################
    # Simulo la cuantificacion del ADC     #
    ########################################
        
        #Normalizo la funcion para que me entre en el rango dinamico
        
        sn = (sn*(ADC.C_max*ADC.q))/(max(sn))

        print('La varianza de la señal contaminada muestreada y normalizada es: ' + str(np.var(sn)))
    
        (t,sq) = ADC.cuantizar(t,sn)        

        print('La varianza de la señal contaminada muestreada, normalizada y cuantficada es: ' + str(np.var(sq)))
    
        e = (sq - sn)

        print('La varianza de la señal de error: ' + str(np.var(e)))

        plt.figure()
        plt.plot(t,e)

    ##########################################
    # Inicializacion del modulo estadistico  #
    ##########################################
    
        STA = sta.statistic()
        
    ##########################################
    # Histograma del error de cuantizacion   #
    ##########################################        
    
        STA.histogram(e)

    ######################################################
    # Inicializo el analizador de espectro               #
    ######################################################
        
        SA = sa.spectrum_analyzer(fs,N)
    
    ##########################################
    # Calculo la PSD del error               #
    ##########################################        
    
        (f,E_PSD) = SA.PSD(e)
        
    ####################################################
    ############# Calculo la PSD de la senal ruidosa    #
    ##################################################        
    
        (f,Sn_PSD) = SA.PSD(sn)        

    ##########################################
    # Calculo la PSD de la senal ruidosa               #
    ####################################################        
    
        (f,Sq_PSD) = SA.PSD(sq)
        
        return (t,f,e,E_PSD)
        
#script

(t,f,e,psd) = testbench()