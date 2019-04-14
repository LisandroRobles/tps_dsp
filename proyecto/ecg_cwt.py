#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:37:47 2019

@author: lisandro

En este script se toma un ECG (de un determinado canal de un registro) y se le
aplica la CWT. Luego se visualiza una porcion del ECG en conjunto con la 
porcion del escalograma

"""

# %% Inclusion de paquetes

#Paquetes para calculos numericos
import numpy as np

#Paquete de wavelets
import pywt

#Paquete para graficar
import matplotlib.pyplot as plt

#Paquetes para manejo de archivos en general
import glob 

#Paquete para manejo de archivos de qtDB
import wfdb

# %% Testbench

def testbench():

    # %% Preparacion del entorno
        
    plt.close('all') #Cierra las ventanas abiertas
    
    # %% Parametros de muestreo
    
    #Frecuencia de muestreo
    fs = 250
    
    #Periodo de muestreo
    Ts = 1/fs
    
    # %% Parametros de registros
    
    chs = 2 #Canales por registro

    registro_inicial = 0 #Registro inicial
    
    registro_final = 0 #Registro final
    
    muestra_inicial = 1000 #Cantidad de muestras por canal

    muestra_final = 2000

    # %% Parametros de cwt
    
    min_scale = 1 #Escala minima
    
    max_scale = 32 #Escala maxima
    
    step = 1 #Paso entre escalas

    scales = np.arange(min_scale,max_scale,step) #Vector de escalas

    #Wavelet madre
    #'cgau1', 'cgau2', 'cgau3','cgau4','cgau5','cgau6','cgau7','cgau8',
    #'cmor',
    #'fbsp',
    #'gaus1','gaus2','gaus3','gaus4','gaus5','gaus6','gaus7','gaus8',
    #'mexh',
    #'morl',
    #'shan'
    wavelet = 'gaus2'

    # %% Lee los registros de ECG de qtDB
        
    paths = glob.glob('/usr/qtdb/*.atr') #Lee el nombre de los registros 
    
    paths = [path[:-4] for path in paths] #Les quita la extension
    
    paths.sort() #Los ordena por orden alfabetico

    # %% Aplica la transformacion a cada registro
    
    #Para cada registro
    for path in paths[registro_inicial:(registro_final + 1)]:
                
        #Leo todos los canales del registro actual
        signals, fields = wfdb.rdsamp(path)

        #Le remuevo la media
        #NO SE PORQUE DEBO HACER ESTO, SI NO NO SE DISTINGEN LOS LATIDOS EN EL ESCALOGRAMA
        signals = signals - np.mean(signals,0)        
        
        #Lee las anotaciones del registro actual
        annotation = wfdb.rdann(path,'atr')   
        
        indices_deteccion = annotation.sample

        #Para cada canal del registro actual        
        for ch in range(chs):
                       
            x = signals[:,ch]
            
            #Aplica la cwt y obtiene la matriz de coeficienes
            cwtmatr,freqs = pywt.cwt(x,scales,wavelet,sampling_period = Ts)
                        
            #Eleva la matriz de coeficientes para obtener el escalograma de potencia
            sc = np.power(cwtmatr,2)
            
            #Libera el espacio de cwtmatr para evitar problemas de memoria
            del cwtmatr
            
            #Normaliza respecto a la potencia total
            sc = np.divide(sc,np.sum(sc))
            
            #Cada coeficiente representa el % de contribucion a la potencia total
            sc = 100*sc
            
            #Almacena solo una parte del registro
            sc = sc[:,muestra_inicial:muestra_final]
            
            vmin = sc.min()
            vmax = sc.max()
            
            #Grafica el segmento en conjunto con la parte del escalograma
            n = np.arange(muestra_inicial,muestra_final)
            peaks = indices_deteccion[np.where((indices_deteccion >= muestra_inicial)&(indices_deteccion < muestra_final))]

            fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)            
            ax[0].plot(n,x[n])
            ax[0].plot(peaks,x[peaks],'x')
            ax[0].set_xlim([muestra_inicial,muestra_final])
            ax[0].grid()
            ax[1].imshow(sc,extent = [muestra_inicial,muestra_final,max_scale,min_scale],cmap = 'plasma',interpolation = 'nearest',aspect = 'auto',vmin = vmin,vmax = vmax)              

                        
# %% Ejecuta el testbench
    
testbench()