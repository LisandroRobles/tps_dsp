#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 00:12:09 2018

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
    
        N = np.power(2,10)       # Muestras
        fs = np.power(2,10)      # Hz
        df = (fs/N)
    
    ########################################
    # Inicializo el generador de funciones #
    ########################################
    
        generador = gen.signal_generator(fs,N)

    ########################################
    # Inicializo el analizador de espectro #
    ########################################
    
        analizador = sa.spectrum_analyzer(fs,N,"fft")
    
    ########################################
    # Genero las senoidales patrones       #
    ########################################    
    
        Ao = 1
        po = 0
        fn1 = 9
        fn2 = 8
    
        #Senoidal patron de 9*Af
        (t,x1) = generador.sinewave(Ao,fn1,po)
                
        #Senoidal patron de 8*Af
        (t,x2) = generador.sinewave(Ao,fn2,po)
        
        fo1 = (fn1*df)
        To1 = 1/fo1
        
        fo2 = (fn2*df)
        To2 = 1/fo2
        
    ########################################
    # Todos los ciclos de fn = 9           #
    ########################################           

        #Calculo del epectro de senoidal patron de 9*Af
        analizador.module_phase(x1)
    
    ########################################
    # Genero 1 solo ciclo de fn = 9        #
    ########################################           
        

    
#        cond = (t < To1).reshape(N,1)
#    
#        x3 = x1*cond
#        
#        #Presentacion temporal de los resultados 
#        plt.figure()
#        plt.title('1 solo ciclo de señaL de fo = ' + str(fo1) +' Hz')
#        plt.plot(t,x3)
#        plt.axis('tight')
#        plt.xlabel('t[s]')
#        plt.ylabel('x(t)[V]')
#        plt.grid()
#        
#        #Calculo del epectro de senoidal patron de 9*Af
#        analizador.module_phase(x3)        

    ##########################################################
    # Genero 1 solo ciclo de fn = 9 demorado 2 ciclos        #
    ##########################################################           
        
#        cond1 = (t >= 2*To1).reshape(N,1)
#        cond2 = (t <= 3*To1).reshape(N,1)
#        cond = cond1*cond2
#    
#        x4 = x1*cond
#        
#        #Presentacion temporal de los resultados estocasticos
#        plt.figure()
#        plt.title('1 solo ciclo de señaL de fo = ' + str(fo1) +' Hz demorado 2 cilos')
#        plt.plot(t,x4)
#        plt.axis('tight')
#        plt.xlabel('t[s]')
#        plt.ylabel('x(t)[V]')
#        plt.grid()
#        
#        #Calculo del espectro
#        analizador.module_phase(x4)

    #############################################################################################
    # Genero 1 solo ciclo de fn = 8 demorado 3 ciclos aprox y uno de fn = 9 al principio        #
    #############################################################################################           

        cond = (t <= To1).reshape(N,1)
    
        x5 = x1*cond
                
        cond1 = (t >= 3*To2).reshape(N,1)
        cond2 = (t <= 4*To2).reshape(N,1)
        cond = cond1*cond2
        
        x6 = x2*cond
        
        x7 = x6 + x5
        
        #Presentacion temporal de los resultados estocasticos
        plt.figure()
        plt.title('1 solo ciclo de señaL de fo = ' + str(fo2) +' Hz y uno de fo = ' + str(fo1) + ' Hz demorada 3 ciclos')
        plt.plot(t,x7)
        plt.axis('tight')
        plt.xlabel('t[s]')
        plt.ylabel('x(t)[V]')
        plt.grid()
        
        #Calculo del espectro
        analizador.module_phase(x7)

    #############################################
    # Igual que lo anterior pero al reves       #
    #############################################           
        
#        cond = (t <= To2).reshape(N,1)
#    
#        x8 = x2*cond
#                
#        cond1 = (t >= 3*To1).reshape(N,1)
#        cond2 = (t <= 4*To1).reshape(N,1)
#        cond = cond1*cond2
#        
#        x9 = x1*cond
#        
#        x10 = x8 + x9
#        
#        #Presentacion temporal de los resultados estocasticos
#        plt.figure()
#        plt.title('1 solo ciclo de señaL de fo = ' + str(fo2) +' Hz y uno de fo = ' + str(fo1) + ' Hz demorada 3 ciclos')
#        plt.plot(t,x10)
#        plt.axis('tight')
#        plt.xlabel('t[s]')
#        plt.ylabel('x(t)[V]')
#        plt.grid()
#        
#        #Calculo del espectro
#        analizador.module_phase(x10)

    #############################################
    # Una de fn = 9 con tres ciclos             #
    #############################################           
                
#        cond = (t <= 3*To1).reshape(N,1)
#        
#        x11 = x1*cond
#                
#        #Presentacion temporal de los resultados estocasticos
#        plt.figure()
#        plt.title('3 solo ciclo de señaL de fo = ' + str(fo1) +' Hz' )
#        plt.plot(t,x11)
#        plt.axis('tight')
#        plt.xlabel('t[s]')
#        plt.ylabel('x(t)[V]')
#        plt.grid()
#        
#        #Calculo del espectro
#        analizador.module_phase(x11)

    ###################################################################
    # Una de fn = 9 con tres ciclos de diferente amplitud             #
    ###################################################################           
                
#        A1 = 0.125
#        A2 = 1
#        A3 = 0.25
#        
#        cond1 = (t < To1).reshape(N,1)
#        cond2 = ((t < 2*To1).reshape(N,1))*((t >= To1).reshape(N,1))
#        cond3 = ((t <= 3*To1).reshape(N,1))*((t >= 2*To1).reshape(N,1))
#        
#        x12 = (A1*x1*cond1) + (A2*x1*cond2) + (A3*x1*cond3)
#                
#        #Presentacion temporal de los resultados estocasticos
#        plt.figure()
#        plt.title('3 solo ciclo de señaL de fo = ' + str(fo1) +' Hz con diferente amplitud' )
#        plt.plot(t,x12)
#        plt.axis('tight')
#        plt.xlabel('t[s]')
#        plt.ylabel('x(t)[V]')
#        plt.grid()
#        
#        #Calculo del espectro
#        analizador.module_phase(x12)

    ###################################################################
    # Una de fn = 9 con tres ciclos de diferente amplitud 3 veces     #
    ###################################################################           
                
        A1 = 0.125
        A2 = 1
        A3 = 0.25
        
        cond1 = (t < To1).reshape(N,1)
        cond2 = ((t < 2*To1).reshape(N,1))*((t >= To1).reshape(N,1))
        cond3 = ((t < 3*To1).reshape(N,1))*((t >= 2*To1).reshape(N,1))

        cond4 = ((t < 4*To1).reshape(N,1))*((t >= 3*To1).reshape(N,1))
        cond5 = ((t < 5*To1).reshape(N,1))*((t >= 4*To1).reshape(N,1))
        cond6 = ((t < 6*To1).reshape(N,1))*((t >= 5*To1).reshape(N,1))

        cond7 = ((t < 7*To1).reshape(N,1))*((t >= 6*To1).reshape(N,1))
        cond8 = ((t < 8*To1).reshape(N,1))*((t >= 7*To1).reshape(N,1))
        cond9 = ((t <= 9*To1).reshape(N,1))*((t >= 8*To1).reshape(N,1))
        
        x13 = (A1*x1*cond1) + (A2*x1*cond2) + (A3*x1*cond3) + (A1*x1*cond4) + (A2*x1*cond5) + (A3*x1*cond6) + (A1*x1*cond7) + (A2*x1*cond8) + (A3*x1*cond9)
#                
#        #Presentacion temporal de los resultados estocasticos
#        plt.figure()
#        plt.title('3 solo ciclo de señaL de fo = ' + str(fo1) +' Hz con diferente amplitud repetido 3 veces' )
#        plt.plot(t,x13)
#        plt.axis('tight')
#        plt.xlabel('t[s]')
#        plt.ylabel('x(t)[V]')
#        plt.grid()
#        
#        #Calculo el espectro
#        analizador.module_phase(x13)
        
    ###########################################################################
    # Una de fn = 9 con un ciclo al principio y otro desfasado 180 grados     #
    ###########################################################################           
                
        cond1 = (t <= To1).reshape(N,1)
        cond2 = ((t <= 2*To1).reshape(N,1))*((t > To1).reshape(N,1))
        
        x14 = (x1*cond1) + (-x1*cond2)
#                
#        #Presentacion temporal de los resultados estocasticos
#        plt.figure()
#        plt.title('3 solo ciclo de señaL de fo = ' + str(fo1) +' Hz con diferente amplitud repetido 3 veces' )
#        plt.plot(t,x14)
#        plt.axis('tight')
#        plt.xlabel('t[s]')
#        plt.ylabel('x(t)[V]')
#        plt.grid()
#        
#        #Calculo el espectro
#        analizador.module_phase(x14)
            
#Script
    
testbench()
