#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 21:08:11 2018

@author: lisandro
"""

# Módulos para Jupyter


#Paquetes

#Warning
import warnings
warnings.filterwarnings('ignore')

#Paquete numerico
import numpy as np

#Paquete de graficos
import matplotlib as mpl
import matplotlib.pyplot as plt

#Paquetes de dsp
import scipy.signal as sig

#Paquete para importar ECG
import scipy.io as sio

import pdsmodulos.spectrum_analyzer as sa

fig_sz_x = 14
fig_sz_y = 13
fig_dpi = 80 # dpi

fig_font_family = 'Ubuntu'
fig_font_size = 16

#Testbench

def testbench():
    
    #Parametros del muestreo
    fs = 1000
    fny = fs/2
    
    #Diseno de filtros
    
    #Plantilla de diseño
    
    #Frecuencias relevantes [Hz]​
    #(Dependen del tipo de filtro!!)
    ws1 = 0.2     #Frecuencia de stop inferior
    wp1 = 0.6    #Frecuencia de paso inferior
    wp2 = 5    #Frecuencia de paso superior
    ws2 = 10   #Frecuencia de stop superior
    
    #Las normaliza respecto del filtro
    ws1 = ws1/(fny)
    wp1 = wp1/(fny)
    wp2 = wp2/(fny)
    ws2 = ws2/(fny)
    freqs = np.array([0,ws1,wp1,wp2,ws2,1],dtype = float)
    
    
    #Magnitud del filtro
    ripple = 0.5 #Ripple en la banda de paso [dB]
    att = 40.0 #Atenuacion en la banda de corte [dB]
    gains = np.array([-att,-att,-ripple,-ripple,-att,-att],dtype = float)
    gains = np.power(10,gains/20)
    
    #Cantidad de coeficientes(largo de la respuesta en frecuencia)
    coefs = 2001
    
    #FIR
    #Disenia FIR por el metodo de ventanas
    #Hay que pasarle un vector que vaya de 0 a 1 con frecuencias relevantes
    #Y un vector con el valor de la respuesta a esas frecuencia
    b_fir = sig.firwin2(coefs,freqs,gains,window = 'hann')
    a_fir = 1
    
    #Calcula la respuesta en frecuencia de los filtros
    w,h_fir = sig.freqz(b_fir,a_fir)

    #Grafica la respuesta en frecuencia de los filtros
    plt.figure()
    plt.title('Respuesta en frecuencia de los filtros')
    plt.plot(w,20*np.log10(np.abs(h_fir)))
    plt.grid()

#Script
    
testbench()