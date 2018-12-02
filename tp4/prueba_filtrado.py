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

    #Parametros de muestreo
    fs = 1000
    N = 20000

    ecg_sinInt = np.zeros((N,3),dtype = float)
    ecg_sinInt[:,0] = ecg_lead[sinInt1,0]
    ecg_sinInt[:,1] = ecg_lead[sinInt2,0]
    ecg_sinInt[:,2] = ecg_lead[sinInt3,0]

    ecg_conInt = np.zeros((N,3),dtype = float)
    ecg_conInt[:,0] = ecg_lead[conIntAlta1,0]
    ecg_conInt[:,1] = ecg_lead[conIntAlta2,0]
    ecg_conInt[:,2] = ecg_lead[conIntAlta3,0]

    #Se plantea la plantilla para los filtros
    #Frecuencias relevantes
    fny = fs/2
    ws1 = 0.2
    wp1 = 0.6
    wp2 = 5
    ws2 = 10
    
    ws1 = ws1/(fny)
    wp1 = wp1/(fny)
    wp2 = wp2/(fny)
    ws2 = ws2/(fny)

    #Atenuaciones [dB]
    Ripple = 0.5
    Att = 15 #Es la mitad ya que se usara filtfilt

    #Creo los dos filtros IIR
    bpf_sos_butter = sig.iirdesign(wp=np.array([wp1, wp2]), ws=np.array([ws1, ws2]), gpass = Ripple, gstop = Att, analog=False, ftype='butter', output='sos')
    bpf_sos_cheby = sig.iirdesign(wp=np.array([wp1, wp2]), ws=np.array([ws1, ws2]), gpass = Ripple, gstop = Att, analog=False, ftype='cheby1', output='sos')

    #Creo los dos filtros FIR
    freqs = np.array([0,ws1,wp1,wp2,ws2,1],dtype = float)
    gains = np.array([-Att,-Att,-Ripple,-Ripple,-Att,-Att],dtype = float)
    gains = np.power(10,gains/20)
    coefs = 1001
    win = 'hann'
    
    b_fir1 = sig.firwin2(coefs,freqs,gains,window = win)

    win = 'flattop'
    coefs = 1001
    b_fir2 = sig.firwin2(coefs,freqs,gains,window = win)

    #Aplico el filtro a los datos con interferencia
    ecg_conInt_Butter = sig.sosfiltfilt(bpf_sos_butter,ecg_conInt,axis = 0)
    ecg_conInt_Cheby = sig.sosfiltfilt(bpf_sos_cheby,ecg_conInt,axis = 0)
    ecg_conInt_Fir1 = sig.filtfilt(b_fir1,1,ecg_conInt,axis = 0)
    ecg_conInt_Fir2 = sig.filtfilt(b_fir2,1,ecg_conInt,axis = 0)

    ecg_sinInt_Butter = sig.sosfiltfilt(bpf_sos_butter,ecg_sinInt,axis = 0)
    ecg_sinInt_Cheby = sig.sosfiltfilt(bpf_sos_cheby,ecg_sinInt,axis = 0)
    ecg_sinInt_Fir1 = sig.filtfilt(b_fir1,1,ecg_sinInt,axis = 0)
    ecg_sinInt_Fir2 = sig.filtfilt(b_fir2,1,ecg_sinInt,axis = 0)
    
    #Para cada senial con interferencia verifico que funcione el filtro
    #Deberia asegurar un nivel isoelectrico cercano a cero
    plt.figure()
    plt.title('Con Int1')
    plt.plot(ecg_conInt[:,0],label = 'Sin Filtrar')
    plt.plot(ecg_conInt_Butter[:,0],label = 'Butter')
    plt.plot(ecg_conInt_Cheby[:,0],label = 'Cheby')
    plt.plot(ecg_conInt_Fir1[:,0],label = 'Fir1')
    plt.plot(ecg_conInt_Fir2[:,0],label = 'Fir2')
    axes_hdl = plt.gca()
    axes_hdl.legend()
    plt.grid()    
    plt.show()

    plt.figure()
    plt.title('Con Int2')
    plt.plot(ecg_conInt[:,1],label = 'Sin Filtrar')
    plt.plot(ecg_conInt_Butter[:,1],label = 'Butter')
    plt.plot(ecg_conInt_Cheby[:,1],label = 'Cheby')
    plt.plot(ecg_conInt_Fir1[:,1],label = 'Fir1')
    plt.plot(ecg_conInt_Fir2[:,1],label = 'Fir2')
    axes_hdl = plt.gca()
    axes_hdl.legend()
    plt.grid()    
    plt.show()

    plt.figure()
    plt.title('Con Int3')
    plt.plot(ecg_conInt[:,2],label = 'Sin Filtrar')
    plt.plot(ecg_conInt_Butter[:,2],label = 'Butter')
    plt.plot(ecg_conInt_Cheby[:,2],label = 'Cheby')
    plt.plot(ecg_conInt_Fir1[:,2],label = 'Fir1')
    plt.plot(ecg_conInt_Fir2[:,2],label = 'Fir2')
    axes_hdl = plt.gca()
    axes_hdl.legend()
    plt.grid()    
    plt.show()
    
    #Para cada senial sin interferencia verifico que NO actue el filtro
    plt.figure()
    plt.title('Sin Int1')
    plt.plot(ecg_sinInt[:,0],label = 'Sin Filtrar')
    plt.plot(ecg_sinInt_Butter[:,0],label = 'Butter')
    plt.plot(ecg_sinInt_Cheby[:,0],label = 'Cheby')
    plt.plot(ecg_sinInt_Fir1[:,0],label = 'Fir1')
    plt.plot(ecg_sinInt_Fir2[:,0],label = 'Fir2')
    axes_hdl = plt.gca()
    axes_hdl.legend()
    plt.grid()    
    plt.show()

    plt.figure()
    plt.title('Sin Int2')
    plt.plot(ecg_sinInt[:,1],label = 'Sin Filtrar')
    plt.plot(ecg_sinInt_Butter[:,1],label = 'Butter')
    plt.plot(ecg_sinInt_Cheby[:,1],label = 'Cheby')
    plt.plot(ecg_sinInt_Fir1[:,1],label = 'Fir1')
    plt.plot(ecg_sinInt_Fir2[:,1],label = 'Fir2')
    axes_hdl = plt.gca()
    axes_hdl.legend()
    plt.grid()    
    plt.show()

    plt.figure()
    plt.title('Sin Int3')
    plt.plot(ecg_sinInt[:,2],label = 'Sin Filtrar')
    plt.plot(ecg_sinInt_Butter[:,2],label = 'Butter')
    plt.plot(ecg_sinInt_Cheby[:,2],label = 'Cheby')
    plt.plot(ecg_sinInt_Fir1[:,2],label = 'Fir1')
    plt.plot(ecg_sinInt_Fir2[:,2],label = 'Fir2')
    axes_hdl = plt.gca()
    axes_hdl.legend()
    plt.grid()    
    plt.show()
    
#Script
    
testbench()