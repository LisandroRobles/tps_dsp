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

    #Levanto el archivo .mat
    mat_struct = sio.loadmat('./ECG_TP4.mat')
    
    #Levanto el electrocardiograma entero
    ecg_lead = mat_struct["ecg_lead"]
    ecg_lead = ecg_lead.reshape(np.size(ecg_lead),1)
        
    #Grafico el electrocardiograma entero
    plt.figure()
    plt.plot(ecg_lead)
    plt.grid()
    plt.title('Electrocardiograma entero')
    plt.axis('tight')
        
    #Se determinan  zonas en las que considero no hay interferencias
    #ya que el nivel isoeletrico se mantiene relativamente constante
    sinInt1 = np.arange(30000,50000,dtype = int)
    sinInt2 = np.arange(400000,420000,dtype = int)
    sinInt3 = np.arange(420000,440000,dtype = int)
    
    plt.figure()
    plt.title('Señales sin interferencia')
    plt.plot(ecg_lead[sinInt1,0],label = 'SinInt1')
    plt.plot(ecg_lead[sinInt2,0],label = 'SinInt2')
    plt.plot(ecg_lead[sinInt3,0],label = 'SinInt3')
    axes_hdl = plt.gca()
    axes_hdl.legend()
    
    
#    #Se determinan las zonas en las que considero hay interferencia de baja
#    #frecuencia, ya que el nivel isoelectrico presenta cierta variacion
#    conIntBaja1 = np.arange(600000,620000,dtype = int)
#    conIntBaja2 = np.arange(840000,860000,dtype = int)
#    conIntBaja3 = np.arange(1100000,1120000,dtype = int)
#
#    plt.figure()
#    plt.title('Señales con interferencia de baja frecuencia')
#    plt.plot(ecg_lead[conIntBaja1,0],label = 'ConIntBaja1')
#    plt.plot(ecg_lead[conIntBaja2,0],label = 'ConIntBaja2')
#    plt.plot(ecg_lead[conIntBaja3,0],label = 'ConIntBaja3')
#    axes_hdl = plt.gca()
#    axes_hdl.legend()    
    
    #Se determinan las zonas en donde considero hay interferencias de alta
    #frecuencia ya que hay unavariaion abrupta en el nivel isoelectrico
    conIntAlta1 = np.arange(720000,740000,dtype = int)
    conIntAlta2 = np.arange(100000,120000,dtype = int)
    conIntAlta3 = np.arange(240000,260000,dtype = int)

    plt.figure()
    plt.title('Señales con interferencia de baja frecuencia')
    plt.plot(ecg_lead[conIntAlta1,0],label = 'ConIntBaja1')
    plt.plot(ecg_lead[conIntAlta2,0],label = 'ConIntBaja2')
    plt.plot(ecg_lead[conIntAlta3,0],label = 'ConIntBaja3')
    axes_hdl = plt.gca()
    axes_hdl.legend()      

    #Se analizara los espectros con y sin interferencia de forma de
    #determinar la plantilla para los filtros
    
    #Parametros de muestreo
    fs = 1000
    N = 20000
        
    #Parametros de Welch
    cant_bloques = 10
    win = 'bartlett'
    
    #Obtengo la psd "promedio" sin interferencia por Welch
    ecg_sinInt = np.zeros((N,3),dtype = float)
    ecg_sinInt[:,0] = ecg_lead[sinInt1,0]
    ecg_sinInt[:,1] = ecg_lead[sinInt2,0]
    ecg_sinInt[:,2] = ecg_lead[sinInt3,0]

    
    (f,S_sinInt,Sv) = sa.welch(ecg_sinInt,fs,k = cant_bloques,window = win,ensemble = True)

    #Obtengo la psd promedio con interferencia por Welch
    ecg_conInt = np.zeros((N,3),dtype = float)
    ecg_conInt[:,0] = ecg_lead[conIntAlta1,0]
    ecg_conInt[:,1] = ecg_lead[conIntAlta2,0]
    ecg_conInt[:,2] = ecg_lead[conIntAlta3,0]      
    
    (f,S_conInt,Sv) = sa.welch(ecg_conInt,fs,k = cant_bloques,window = win,ensemble = True)
    
    #Grafico las psd promedio de ambas zonas
    
    plt.figure()
    plt.title('PSD de señales sin interferencia')
    plt.plot(f,10*np.log10(S_sinInt[:,0]),label = ['SinInt1'])
    plt.plot(f,10*np.log10(S_sinInt[:,1]),label = ['SinInt2'])
    plt.plot(f,10*np.log10(S_sinInt[:,2]),label = ['SinInt3'])
    plt.plot(f,10*np.log10(S_conInt[:,0]),label = ['ConIntAlta1'])
    plt.plot(f,10*np.log10(S_conInt[:,1]),label = ['ConIntAlta2'])
    plt.plot(f,10*np.log10(S_conInt[:,2]),label = ['ConIntAlta3'])
    axes_hdl = plt.gca()
    axes_hdl.legend()
    plt.show()
    
    #Se puede observar que la senial on interferencia tiene mayores compo
    #nentes en baja frecuencia por debajo de 1Hz
    
    #Para determinar las frecuencias superiores se busca aquellas que permitan
    #contener el 90% de la energia de la senial
    # Calculo donde esta el 90% de la energia de la señal OK
    enTot = np.sum(S_sinInt[:,0])
    en90 = 0.9*enTot
    
    suma = 0
    for i in range(len(S_sinInt[:,0])):
        suma += S_sinInt[i,0]
        if suma >= en90:
            frec90 = f[i]
            break
        
    print(frec90)


    
#Script
    
mat_struct = testbench()