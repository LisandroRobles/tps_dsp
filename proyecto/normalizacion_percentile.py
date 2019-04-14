#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:30:45 2019

@author: lisandro

En este script se obtiene un ECG correspondiente de un canal determinado y de 
un registro determinado. Se detecta el valor de los picos y se grafica un histo
grama de los mismos. Luego, se evalua normalizarlos usando los percentiles 

"""

# %% Inclusion de paquetes

#Paquetes para calculos numericos
import numpy as np

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

    # %% Parametros de registro
       
    registro = 3  #Registro deseado (0:81)

    ch = 0  #Canal deseado (0:1)

    # %% Parametros de histograma
    
    n_bins = 50

    # %% Parametros de normalizacion
    
    p_inf = 20 #Percentile inferior
    
    p_sup = 90 #Percentile superior

    # %% Lee los registros de ECG de qtDB

    paths = glob.glob('/usr/qtdb/*.atr') #Lee el nombre de los registros

    paths = [path[:-4] for path in paths] #Les quita la extension
    
    paths.sort() #Los ordena por orden alfabetico

    # %% Lee el canal seleccionado
    
    signals, fields = wfdb.rdsamp(paths[registro]) #Lee el registro

    x = signals[:,ch] #Lee el canal seleccionado del registro

    # %% Lee las anotaciones

    annotation = wfdb.rdann(paths[registro],'atr') #Obtiene las anotaciones   
    
    peaks = annotation.sample #Obtiene el indice de cada pico    

    # %% Obtiene el valor de los picos
    
    peaks_values = x[peaks]

    # %% Genera un histograma con el ecg
    
    dist = peaks_values
    
    plt.figure()
    plt.hist(dist,bins = n_bins)

    # %% Imprime valores caracteristicos de la distribucion

    print('Valores caracteristicos.\n')
    print('Minimo: {:.2f}.\n'.format(min(dist)))
    print('Maximo: {:.2f}.\n'.format(max(dist)))
    print('Desvio estandar: {:.2f}.\n'.format(np.std(dist)))
    print('Rango: {:.2f}.\n'.format(max(dist)-min(dist)))
    print('Percentil {:.2f}: {:.2f}.\n'.format(p_inf,np.percentile(dist,p_inf)))
    print('Percentil {:.2f}: {:.2f}.\n'.format(p_sup,np.percentile(dist,p_sup)))

# %% Ejecuta el testbench
    
testbench()
