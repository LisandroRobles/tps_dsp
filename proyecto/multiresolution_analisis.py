#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:37:47 2019

@author: lisandro

En este script se realiza analisis multiresolucion sobre un ECG correspondiente
a un canal determinado de un registro determinado 
"""

# %% Inclusion de paquetes

from IPython import get_ipython
get_ipython().magic('reset -sf')

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
    
    #Borra los graficos
    plt.close('all')

    # %% Parametros de registro    

    registro = 4 #Registro deseado

    ch = 0 #Canal deseado

    # %% Parametros de la transformada wavelet

    #Rango sobre el que se aplica la transformacion
    n1 = 10000
    
    n2 = 12000

#    #Wavelet madre
#    
#    #Ya definida en la libreria
#    #'bior1.1','bior1.3','bior1.5',
#    #'bior2.2','bior2.4','bior2.6','bior2.8',
#    #'bior3.1','bior3.3','bior3.5','bior3.7','bior3.9',
#    #'bior4.4',
#    #'bior5.5',
#    #'bior6.8',
#    #'coif1','coif2','coif3','coif4','coif5','coif6','coif7','coif8','coif9',
#    #'coif10','coif11','coif12','coif13','coif14','coif15','coif16','coif17',
#    #'db1','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12',
#    #'db13','db14','db15','db16','db17','db18','db19','db20','db21','db22',
#    #'db23', 'db24','db25','db26','db27','db28','db29','db30','db31','db32',
#    #'db33','db34','db35','db36','db37','db38',
#    #'dmey',
#    #'haar',
#    #'rbio1.1','rbio1.3','rbio1.5',
#    #'rbio2.2','rbio2.4','rbio2.6','rbio2.8',
#    #'rbio3.1','rbio3.3','rbio3.5','rbio3.7','rbio3.9',
#    #'rbio4.4',
#    #'rbio5.5',
#    #'rbio6.8',
#    #'sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11',
#    #'sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20'
#
#    wav = 'db2'
#    
#    w = pywt.Wavelet(wav)

    
#    #Especificando el banco de filtros
#
#    c = np.sqrt(2)/2
#
#    dec_lo, dec_hi, rec_lo, rec_hi = [c, c], [-c, c], [c, c], [c, -c]
#
#    filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
#
#    myWavelet = pywt.Wavelet(name="myHaarWavelet", filter_bank=filter_bank)
#
#    w = myWavelet


   #Especificando el banco de filtros - 2

    rec_lo = [(1/8),(3/8),(3/8),(1/8)]

    rec_hi = [0,2,-2,0]

    dec_lo = [(1/8),(3/8),(3/8),(1/8)]

    dec_hi = [0,-2,2,0]

    filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]

    myWavelet2 = pywt.Wavelet(name="myWavelet", filter_bank=filter_bank)

    w = myWavelet2


    # %% Lee los registros de ECG de qtDB
    
    paths = glob.glob('/usr/qtdb/*.atr') #Lee el nombre de los registros
    
    paths = [path[:-4] for path in paths] #Les quita la extension
    
    paths.sort() #Los ordena por orden alfabetico
    
    # %% Lee el canal seleccionado

    signals, fields = wfdb.rdsamp(paths[registro]) #Lee el registro

    x = signals[0:131072,ch] #Lee el canal seleccionado del registro

    # %% Transformada wavelet discreta - Analisis con banco de filtros  

    #Niveles en los que se descompone
    
    #Obtiene los coeficientes de la descomposicion - Decimando

    #levels = pywt.dwt_max_level(len(x),w)

    #coeffs_dec = pywt.wavedec(x,w,level = levels)

    #SWT - Algoritmo A'Trous (sin decimar)

    #levels = pywt.swt_max_level(len(x))

    levels = 5

    coeffs_undec = pywt.swt(x,w,level = levels)

    # %% Grafica las señales de aproximación - DWT (decimando)     

#    coeffs = coeffs_dec #Coeficientes que se graficaran
#
#    coeffs_aux = []
#    
#    for i in range(len(coeffs)):
#        coeffs_aux.append(coeffs[i])
#
#    fig, ax = plt.subplots(levels + 1, sharex=True)
#        
#    for i in range(levels,-1,-1):
#        for j in range(len(coeffs)):
#            if j > i:
#                coeffs_aux[j] = np.zeros_like(coeffs_aux[j])
#        xi = pywt.waverec(coeffs_aux,w)
#        ax[i].plot(xi)

    # %% Grafica las señales de detalle - DWT (decimando)
    
#    fig, ax = plt.subplots(levels, sharex=True, sharey= False)
#
#    coeffs_aux = list(np.zeros_like(coeffs))
#        
#    for i in range(levels):
#        for j in range(len(coeffs)):
#            if j == (i+1):
#                coeffs_aux[j] = coeffs[j]
#            else:
#                coeffs_aux[j] = np.zeros_like(coeffs[j])
#        xi = pywt.waverec(coeffs_aux,w)
#        ax[i].plot(xi)
#     
#    plt.figure()
#    
#    plt.plot(x)
#
#    plt.grid()    
    
    # %% Grafica las señales de aproximación - SWT (sin decimar)

    coeffs = coeffs_undec #Coeficientes que se graficaran

    fig, ax = plt.subplots(levels + 1, sharex=True)
    
    ax[levels].plot(x[n1:n2])
    for i in range(levels-1,-1,-1):
        xi = coeffs[i][0]
        ax[i].plot(xi[n1:n2])
        
    # %% Grafica las señales de detalle - SWT (sin decimar)

    fig, ax = plt.subplots(levels, sharex=True)
    
    for i in range(levels-1,-1,-1):
        xi = coeffs[i][1]
        ax[i].plot(xi[n1:n2])    
    
# %% Ejecuta el testbench

testbench()