#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 14:43:03 2018

@author: lisandro
"""

#Librerias

import numpy as np
import pandas as pd
from IPython.display import HTML
import matplotlib.pyplot as plt

import pdsmodulos.signal_generator as gen
import pdsmodulos.spectrum_analyzer as sa
import time 

#Testbench

def testbench():
    
    ###########################
    # Parametros del muestreo #
    ###########################
    
        N = (16,32,64,128,256,512,1024,2048)

    ###########################
    # Parametros del muestreo #
    ###########################
        
        tus_resultados = [['--','--']]
        indices = ['Tamaño de la señal N']
    
    ########################################
    # Inicializa el ciclo de prueba        #       
    ########################################
        
        for Ni in N:

        #############################################################
        # Se setea fs de forma de siempre tener resolucion unitaria #
        #############################################################

            fsi = Ni            
            
        ########################################
        # Inicializo el generador de funciones #
        ########################################
        
            generador = gen.signal_generator(fsi,Ni)
                    
        ########################################
        # Funcion de prueba                    #       
        ########################################
        
            Ao = 1
            fo = np.round(Ni/6)
            po = 0
            
            (t,x) = generador.sinewave(Ao,fo,po)
            
        ########################################
        # Inicializo el analizador de espectro #
        ########################################

            analizador = sa.spectrum_analyzer(fsi,Ni,algorithm = "dft")
        
        ########################################
        # Calcula la DFT usando la DFT         #       
        ########################################
    
            startDFT = time.time()
            (X) = analizador.transform(x)
            endDFT = time.time()
            timeDFT = np.around(endDFT - startDFT,decimals = 3)
            print('El tiempo de ejecucion de la DFT es de: ' + str(timeDFT))
            
        ########################################
        # Setea la FFT como algoritmo          #       
        ########################################
        
            analizador.set_algorithm("fft")
            
        ########################################
        # Calcula la DFT usando la FFT         #       
        ########################################
    
            startFFT = time.time()
            (X) = analizador.transform(x)
            endFFT = time.time()
            timeFFT = np.around(endFFT - startFFT,decimals = 6)
            print('El tiempo de ejecucion de la FFT es de: ' + str(timeFFT))

        ################################################
        # Almacena los resultados en una lista         #       
        ################################################
    
            tus_resultados.append([str(timeDFT),str(timeFFT)])
            #tus_resultados.append([timeDFT,timeFFT])

            indices.append(Ni)

    ############################################
    # Ordena los resultados en un frame        #       
    ############################################
        
        df = pd.DataFrame(tus_resultados, columns = ['tiempo transcurrido DFT (s)','tiempo transcurrido FFT (s)'], index = indices)

    #################################################
    # Grafica los resultados en funcion de N        #       
    #################################################
        
        #Para la DFT
        
        tiemposDFT = [float(i[0]) for i in tus_resultados if i[0] != '--']
        
        plt.figure()
        plt.stem(N,tiemposDFT)
        plt.title('Rendimiento de DFT')
        plt.axis('tight')
        plt.xlabel('Largo senal [N]')
        plt.ylabel('t[s]')
        plt.grid()

        tiemposFFT = [float(i[1]) for i in tus_resultados if i[0] != '--']

        plt.figure()
        plt.stem(N,tiemposFFT)
        plt.title('Rendimiento de FFT')
        plt.axis('tight')
        plt.xlabel('Largo senal [N]')
        plt.ylabel('t[s]')
        plt.grid()
    
        #plt.figure()
        #df.plot(y =['tiempo transcurrido DFT (s)','tiempo transcurrido FFT (s)'],kind='bar');
        
#Script
        
testbench()