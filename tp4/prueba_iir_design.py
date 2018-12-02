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
    ws1 = 1     #Frecuencia de stop inferior
    wp1 = 10    #Frecuencia de paso inferior
    wp2 = 50    #Frecuencia de paso superior
    ws2 = 100   #Frecuencia de stop superior
    
    #Las normaliza respecto del filtro
    ws1 = ws1/(fny)
    wp1 = wp1/(fny)
    wp2 = wp2/(fny)
    ws2 = ws2/(fny)
    
    #Magnitud del filtro
    ripple = 0.5 #Ripple en la banda de paso [dB]
    att = 40.0 #Atenuacion en la banda de corte [dB]
    
    #IIR
    #The type of IIR filter to design:
    #Butterworth : ‘butter’
    #Chebyshev I : ‘cheby1’
    #Chebyshev II : ‘cheby2’
    #Cauer/elliptic: ‘ellip’
    #Bessel/Thomson: ‘bessel’

    bpf_sos_butter = sig.iirdesign(wp=np.array([wp1, wp2]), ws=np.array([ws1, ws2]), gpass = ripple, gstop = att, analog=False, ftype='butter', output='sos')
    bpf_sos_cheby = sig.iirdesign(wp=np.array([wp1, wp2]), ws=np.array([ws1, ws2]), gpass = ripple, gstop = att, analog=False, ftype='cheby1', output='sos')
    bpf_sos_cauer = sig.iirdesign(wp=np.array([wp1, wp2]), ws=np.array([ws1, ws2]), gpass = ripple, gstop = att, analog=False, ftype='ellip', output='sos')
    
    #Calcula la respuesta en frecuencia de los filtros
    w,h_butter = sig.sosfreqz(bpf_sos_butter)
    w,h_cheby = sig.sosfreqz(bpf_sos_cheby)
    w,h_cauer = sig.sosfreqz(bpf_sos_cauer)
    
    #Grafica la respuesta en frecuencia de los filtros
    plt.figure()
    plt.title('Respuesta en frecuencia de los filtros')
    plt.plot(w,20*np.log10(np.abs(h_butter)),w,20*np.log10(np.abs(h_cheby)),w,20*np.log10(np.abs(h_cauer)))
    plt.grid()
    plt.legend(['Butter','Cheby','Cauer'])
            
#Script
    
testbench()