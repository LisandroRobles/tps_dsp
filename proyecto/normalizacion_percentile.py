#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:30:45 2019

@author: lisandro

Este script motivado por el problema de normalizar los escalogramas de potencia
a la hora de generar imagenes a partir de los mismos. Para solucionar esto se
han manejado tres alternativas:
    
    1) Se normaliza mediante el minimo y maximo del escalograma de cada cada
    ventana. Esto no era viable ya que aplicar un factor variable por segmento
    distorsionaba la contribucion de cada segmento a la varianza total.
    
    2) Primero se realiza el escalograma de todo el registro y se utiliza como
    factor para cada imagen el minimo y maximo del registro total. Si bien esto
    unifica las escalas de cada imagen, los mismos varian segun el registro que
    se use y debera relizarse una CWT sobre todo el registro en cada iteracion.
    
    3) Utilizar un factor de normalizacion fijo para todos los registros.
    
Para realizar 3) se computa el maximo del escalograma de potencia de cada 
registro y se toma un percentil del mismo. El mismo debe ser lo suficientemente
grande como para que sature la menor cantidad de registros, pero no tan grande
para que los latidos de los escalogramas con maximos mas chicos se puedan 
distinguir.

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

#Paquete de wavelets
import pywt

# %% Testbench

def testbench():
    
    # %% Preparacion del entorno
        
    plt.close('all') #Cierra las ventanas abiertas

    # %% Parametros de la transformada wavelet
    
    min_scale = 1 # Escala minima
    
    max_scale = 51 # Escala maxima
    
    step = 1 # Paso entre escalas
    
    scales = np.arange(min_scale,max_scale,step) # Vector de escalas
    
    fs = 250 # Frecuencia de muestreo
    
    Ts = 1/fs # Periodo de muestreo 
    
    # Wavelet madre
    #'cgau1', 'cgau2', 'cgau3','cgau4','cgau5','cgau6','cgau7','cgau8',
    #'cmor',
    #'fbsp',
    #'gaus1','gaus2','gaus3','gaus4','gaus5','gaus6','gaus7','gaus8',
    #'mexh',
    #'morl',
    #'shan'
    wavelet = 'gaus2'
    
    # %% Parametros de almacenamiento
    
    filename = 'normalizacion/normalizacion1'

    # %% Parametros de histograma
    
    nbins = 500 #Cantidad de bins del histograma

    # %% Parametros de normalizacion
    
    p_min = 50 # Percentil minimo que se computa
    
    p_step = 5 # Paso entre percentiles
    
    p_max = 100 # Percentil maximo que se computa 
    
    percentiles = np.arange(p_min,p_max,p_step) #Percentiles que se computan

    # %% Lee los registros de ECG de qtDB

    paths = glob.glob('/usr/qtdb/*.atr') #Lee el nombre de los registros

    paths = [path[:-4] for path in paths] #Les quita la extension
    
    paths.sort() #Los ordena por orden alfabetico

    # %% Lee todos los registros
    
    vmax = [] # Lista donde se almacenara el maximo de cada escalograma
        
    for registro in range(len(paths)): # Para cada registro
        
        signals, fields = wfdb.rdsamp(paths[registro]) # Lee el registro
        
        for canal in range(np.size(signals,1)): # Para cada derivacion
        
            x = signals[:,canal] # Lee el canal actual del registro actual

            x = x - np.mean(x) # Media 0
            
            x = x/np.std(x) #Normaliza la potencia de x

            # Aplica la cwt y obtiene la matriz de coeficientes
            cwtmatr,freqs = pywt.cwt(x,scales,wavelet,sampling_period = Ts) 
            
            aux,freqs = pywt.cwt(x[0:100],scales,wavelet,sampling_period = Ts) 
                
            sc = np.power(cwtmatr,2) # Obtiene el escalograma de potencia

            del cwtmatr # Libera el espacio de cwtmatr

            for i in range(len(scales)): # Se normaliza en escala
        
                sc[i,:] = np.divide(sc[i,:],scales[i])
                
            sc = np.divide(sc,np.var(x)) # Normaliza respecto a la energia
                                
            vmax.append(sc.max()) # Parametro minimo para normalizar la imagen

            del sc # Libera el espacio de sc

        # Informa que termino de procesar las derivaciones del registro actual
        print('Procesado registro: {:d}.\n'.format(registro))
        
    # %% Analiza los resultados
    
    print('\nParámetros de los máximos.\n')
    
    print('Máximo: {:.4f}\n'.format(max(vmax)))
    
    print('Mínimo: {:.4f}\n'.format(min(vmax)))
    
    print('Valor medio: {:.4f}\n'.format(np.mean(vmax)))
    
    print('Desviación estándar: {:.4f}\n'.format(np.std(vmax)))
    
    for i in range(len(percentiles)):
    
        print('Percentil {:d}: {:.4f}\n'.format(percentiles[i]\
              ,np.percentile(vmax,percentiles[i])))

    # %% Grafica los histogramas
    
    print('\nHistograma de los máximos.\n') # Histograma de los máximos
    
    plt.figure(1)
    
    plt.hist(vmax,bins = nbins)
    
    plt.axis('tight')
    
    plt.grid()

    plt.savefig('{:s}.png'.format(filename))
    
    plt.show()

    # %% Almacena resultados en txt
    
    file = open('{:s}.txt'.format(filename),"w") # Abre el archivo para escritura
    
    file.write('Parámetros de wavelet.\n\n')
    
    file.write('Wavelet: {:s}\n'.format(wavelet))
    
    file.write('Escalas: {:d}:{:.2f}:{:d}\n'.format(min_scale,step,max_scale))
    
    file.write('\nParámetros de los máximos.\n\n')

    file.write('Máximo: {:.4f}\n'.format(max(vmax)))
    
    file.write('Mínimo: {:.4f}\n'.format(min(vmax)))
    
    file.write('Valor medio: {:.4f}\n'.format(np.mean(vmax)))
    
    file.write('Desviación estándar: {:.4f}\n'.format(np.std(vmax)))
    
    for i in range(len(percentiles)):
        
        file.write('Percentil {:d}: {:.4f}\n'.format(percentiles[i]\
                   ,np.percentile(vmax,percentiles[i])))    

    file.write('\nParámetros de histograma.\n\n')

    file.write('Cantidad de bins: {:d}'.format(nbins))
    
# %% Ejecuta el testbench
    
testbench()
