#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:33:15 2018

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
        
    #Parametros del muestreo
    N = np.array([2000], dtype = int)
    
    #Frecuencias de muestreo
    fs = 1000
    
    #Cantidad de realizaciones
    S = 500
    
    #En cuantos bloques divido
    L = N/10
    
    #Overlap entre bloques
    ovi = 50
    
    #Ventana utilizada
    wi = 'bartlett'
    
    #Aca se almacenaran los resultados
    tus_resultados_b = []
    tus_resultados_w = []
        
    #Para cada largo de señal
    for Ni in N:
                
        #Enciendo el generador de funciones
        generador = gen.signal_generator(fs,Ni)
                
        #Lista en que alamacenre las distribuciones para cada realizacion
        dist = []
        
        #Distribucion elegida para cada realizacion (todas normales)
        for i in range(0,S):
            dist.append("normal")
        
        #Media - Todas las realizaciones de media 0
        u = np.zeros((S,1),dtype = float)
        
        #Varianza - Todas las realizaciones de desvio estandar de raiz de 2
        var = 1
        s = np.sqrt(var)*np.ones((S,1),dtype = float)
        
        #Llamo al metodo que genera ruido blanco
        #Genera una matriz de NxS, donde N = Filas y S = Columnas
        (t,x) = generador.noise(dist,u,s)
            
        #Realizo de forma matricial el modulo del espectro de todas las realizaciones
        (f,Sxm_w,Sxv_w) = sa.welch(x,fs,L,window = wi,overlap = ovi)
        
        (f,Sxm_b,Sxv_b) = sa.bartlett(x,fs,nsect = 10)
        
        #Calculo el area de ese espectro "promedio"
        #El area de la psd da la potencia
        valor_esperado = (np.mean(Sxm_w))
        print('Valor esperado:' + str(valor_esperado))
        sesgo = valor_esperado - np.power(s[0,0],2)

        #Calculo el area de eso
        varianza = (np.mean(Sxv_w))
        print('Varianza del estimador:' + str(varianza))
        
        #Almaceno los resultados para esta largo de señal
        tus_resultados_w.append([str(sesgo),str(varianza)])

        #Calculo el area de ese espectro "promedio"
        #El area de la psd da la potencia
        valor_esperado = (np.mean(Sxm_b))
        print('Valor esperado:' + str(valor_esperado))
        sesgo = valor_esperado - np.power(s[0,0],2)

        #Calculo el area de eso
        varianza = (np.mean(Sxv_b))
        print('Varianza del estimador:' + str(varianza))
        
        #Almaceno los resultados para esta largo de señal
        tus_resultados_b.append([str(sesgo),str(varianza)])
        
    #Almaceno el resultado en el dataframe
    df_w = DataFrame(tus_resultados_w, columns=['$s_P$', '$v_P$'],index=N)
    
    df_b = DataFrame(tus_resultados_b, columns=['$s_P$', '$v_P$'],index=N)
    
    print(df_b)
    
    print(df_w)
    
#Script

testbench()