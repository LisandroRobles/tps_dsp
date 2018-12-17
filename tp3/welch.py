# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:29:45 2018

@author: Francisco
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft
from numpy import linalg as LA

def barlett(N):
    
    k = np.linspace(0, (N-1), N)
    w = (2/(N-1))*((N-1)/2 - np.abs(k-(N-1)/2))
    return w

def testbench():

    fs = 1000   # Frecuencia de muestreo
    N = np.linspace(30, 5000, 498)
#    Ns = np.array([10000], dtype=np.float)
    K = 10          # Cantidad de bloques
    overlap = 50    # Overlap en %
    
    # Parametros del ruido blanco gaussiano
    mean = 0
    variance = 2
     
    result_sesgo = []
    result_var = []

#%% Variamos N
    
    for n in N:
        
        L = int(n/K) #Largo de cada bloque
        D = int(L * overlap/100) # Offset de los bloques
        cant_promedios = 1 + int((n-L)/D)   # Cantidad de promedios realizados
        
        window = barlett(L)
        window = window/LA.norm(window)
        
        real_bin_psd = variance/L
        noise = np.random.normal(mean, np.sqrt(variance), int(n))
    
        n1 = 0
        psd_average = 0
        for i in range(cant_promedios):
            noise_spectrum = fft(noise[n1:(n1+L)]*window, axis = 0)
            psd = pow((1/L)*np.abs(noise_spectrum), 2)
            psd_average = psd_average + psd/cant_promedios
            n1 = n1 + D
            
        psd_average = psd_average*L # NO TENGO IDEA DE DONDE SALE ESTE *L
        
        # Varianza del estimador 
        bin_var = np.var(psd_average)*L**2
        
        # Valor esperado de la PSD
        bin_PSD_esperado = np.average(psd_average)
    
        # Varianza de la señal, area del PSD
        varianza = np.sum(psd_average)
    
        sesgo = np.abs(real_bin_psd - bin_PSD_esperado)
        
        result_sesgo.append(sesgo)
        result_var.append(bin_var)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(N, result_sesgo)
    plt.title("Sesgo en funcion de N")
    plt.xlabel("N")
    plt.ylabel("Sesgo")
    plt.yscale("log")
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.plot(N, result_var)
    plt.title("Varianza en funcion de N")
    plt.xlabel("N")
    plt.ylabel("Varianza")
#    plt.yscale("log")
    plt.grid()
    
#    print("Varianza del bin promedio: " + str(bin_var))
#    print("Valor esperado del bin: " + str(bin_PSD_esperado))
#    print("Area: " + str(varianza))
#    print("Sesgo: " + str(sesgo))
    
#%% Variamos K
    
    K = np.array([1, 2, 5, 10, 20, 25, 40, 50, 100], dtype=np.float)
    K = np.linspace(1, 200, 200)
    N = 1000
    
    result_sesgo = []
    result_var = []
    
    df = []
    
    for k in K:
        
        L = int(N/k)
        D = int(L * overlap/100) # Offset de los bloques
        cant_promedios = 1 + int((N-L)/D)   # Cantidad de promedios realizados
        
        df.append(fs/L)
        
        window = barlett(L)
        window = window/LA.norm(window)
        
        real_bin_psd = variance/L
        noise = np.random.normal(mean, np.sqrt(variance), int(N))
    
        n1 = 0
        psd_average = 0
        for i in range(cant_promedios):
            noise_spectrum = fft(noise[n1:(n1+L)]*window, axis = 0)
            psd = pow((1/L)*np.abs(noise_spectrum), 2)
            psd_average = psd_average + psd/cant_promedios
            n1 = n1 + D
            
        psd_average = psd_average*L # NO TENGO IDEA DE DONDE SALE ESTE *L
        
        # Varianza del estimador 
        bin_var = np.var(psd_average)*L**2
        
        # Valor esperado de la PSD
        bin_PSD_esperado = np.average(psd_average)
    
        # Varianza de la señal, area del PSD
        varianza = np.sum(psd_average)
    
        sesgo = np.abs(real_bin_psd - bin_PSD_esperado)
        
        result_sesgo.append(sesgo)
        result_var.append(bin_var)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(K, df)
    plt.title("Resolucion espectral en funcion de K")
    plt.xlabel("K")
    plt.ylabel("$\Delta f$")
#    plt.yscale("log")
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.plot(K, result_var)
    plt.title("Varianza en funcion de K")
    plt.xlabel("K")
    plt.ylabel("Varianza")
    plt.yscale("log")
    plt.grid()
    
#%% Variamos overlap
    
    N = 1000
    K= 10
    overlap = np.linspace(10, 50, 5)
#    overlap = np.array([10, 50], dtype=np.float)
    L = int(N/K)
    
    result_sesgo = []
    result_var = []
    
    df = []
    
    noise = np.random.normal(mean, np.sqrt(variance), int(N))
    
    for OL in overlap:
        
        D = int(L * OL/100) # Offset de los bloques
        cant_promedios = 1 + int((N-L)/D)   # Cantidad de promedios realizados
        
        df.append(fs/L)
        
        window = barlett(L)
        window = window/LA.norm(window)
        
        real_bin_psd = variance/L
    
        n1 = 0
        psd_average = 0
        for i in range(cant_promedios):
            noise_spectrum = fft(noise[n1:(n1+L)]*window, axis = 0)
            psd = pow((1/L)*np.abs(noise_spectrum), 2)
            psd_average = psd_average + psd/cant_promedios
            n1 = n1 + D
            
        psd_average = psd_average*L # NO TENGO IDEA DE DONDE SALE ESTE *L
        
        # Varianza del estimador 
        bin_var = np.var(psd_average)*L**2
        
        # Valor esperado de la PSD
        bin_PSD_esperado = np.average(psd_average)
    
        # Varianza de la señal, area del PSD
        varianza = np.sum(psd_average)
    
        sesgo = np.abs(real_bin_psd - bin_PSD_esperado)
        
        result_sesgo.append(sesgo)
        result_var.append(bin_var)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(overlap, df)
    plt.title("Resolucion espectral en funcion del overlap")
    plt.xlabel("Overlap [%]")
    plt.ylabel("$\Delta f$")
#    plt.yscale("log")
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.plot(overlap, result_var)
    plt.title("Varianza en funcion del overlap")
    plt.xlabel("Overlap [%]")
    plt.ylabel("Varianza")
#    plt.yscale("log")
    plt.grid()
    
    return result_var

result_var = testbench()