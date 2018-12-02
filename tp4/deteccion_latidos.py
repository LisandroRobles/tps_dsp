#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 21:08:11 2018

@author: lisandro
"""
#Paquetes

#Paquete numerico
import numpy as np

#Paquete de graficos
import matplotlib.pyplot as plt

#Paquetes de dsp
import scipy.signal as sig

#Paquete para importar ECG
import scipy.io as sio

import pdsmodulos.spectrum_analyzer as sa

#Testbench

def testbench():

    mat_struct = sio.loadmat('./ECG_TP4.mat')
    
    #Levanto el electrocardiograma entero
    ecg_lead = mat_struct["ecg_lead"]
    ecg_lead = ecg_lead.reshape(np.size(ecg_lead),1)
    
    print(np.mean(ecg_lead))
    
    ecg_lead = ecg_lead - np.mean(ecg_lead)
    
    print(np.mean(ecg_lead))
    
    #Lograr estimar una interferente
    
    #Grafico el electrocardiograma entero
    plt.figure()
    plt.plot(ecg_lead)
    plt.grid()
    plt.title('Electrocardiograma entero')
    plt.axis('tight')
    
    #Levanto los indices de los latidos
    indices_latidos = mat_struct["qrs_detections"]
    indices_latidos = indices_latidos.reshape(np.size(indices_latidos),1)
        
    #Levanto el patron de los latidos
    largo_ventana = 800

    #Matriz donde almacenare todos los latidos
    latidos = np.zeros((np.size(indices_latidos,0),largo_ventana))

    j = 0
    for j in range(0,np.size(indices_latidos,0)):
        latidos[j,:] = ecg_lead[(int(indices_latidos[j,0]) - int(((largo_ventana/2)))):((int(indices_latidos[j,0]) + int((largo_ventana/2)))),:]
    
    #Grafico eso
    plt.figure()
    plt.plot(latidos[0,:])
    plt.grid()
    plt.title('Sin interferencia (aprox)')
    plt.axis('tight')
    
#    analizador = sa.spectrum_analyzer(fs,N)
    
#    ecg_sin_interferencia = ecg_sin_interferencia - np.mean(ecg_sin_interferencia)
#    (f,Sxx) = analizador.psd(ecg_sin_interferencia,xaxis = 'phi')
#    
#    #Grafico la psd de eso
#    plt.figure()
#    plt.plot(f,Sxx/np.max(Sxx))
#    plt.title('Espectro (sin interferencia)')
#    plt.axis('tight')
#    plt.grid()
    
    return mat_struct
    

    
#Script
    
mat_struct = testbench()