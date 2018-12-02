#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 22:26:41 2018

@author: lisandro
"""

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

#Paquete para interpolar
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import correlate

import pdsmodulos.spectrum_analyzer as sa
import pdsmodulos.filters as filt
import statistics as st

#Paquete de manejo de datos
import pandas as pd

#Medicion de tiempo
import time 

#Testbench

def testbench():
    
    #Levanto el archivo .mat
    mat_struct = sio.loadmat('./ECG_TP4.mat')
    
    #Levanto el electrocardiograma entero
    ecg_lead = mat_struct["ecg_lead"]
    ecg_lead = np.array(ecg_lead,dtype = float)
    ecg_lead = ecg_lead.reshape(np.size(ecg_lead),1)
        
    #Deteccion de R por otros medios
    qrs_detections = mat_struct["qrs_detections"]
    
    #Latidos patrones
    d1 = mat_struct["heartbeat_pattern1"]
    d2 = mat_struct["heartbeat_pattern2"]

    #Patron QRS
    qrs_pattern1 = mat_struct["qrs_pattern1"]
    
    #Preprocesamiento
    #Para detetar bien el ecg debe estar planito
    #Filtro utilizando el filtro de mediana con submuestreo
    fs1 = 1000
    fs2 = 200
    (t,ecg_lead_diezmado) = filt.decimate(ecg_lead,fs1,fs2)
    ecg_lead_med200 = sig.medfilt(ecg_lead_diezmado,[int(0.2*fs2)+1,1])
    ecg_lead_med600_con_diezmado = sig.medfilt(ecg_lead_med200,[int(0.6*fs2)+1,1])
    B_con_diezmado = np.zeros((np.size(ecg_lead,0),1),dtype = float)
    B_con_diezmado[0:np.size(ecg_lead,0):int(fs1/fs2)] = ecg_lead_med600_con_diezmado
    
    x = ecg_lead - B_con_diezmado

    #Verifico que se haya invertido en el tiempo
    plt.figure()
    plt.title('ECG filtrado con filtro de mediana')
    plt.plot(ecg_lead)
    plt.plot(B_con_diezmado)
    plt.grid() 
    plt.show()
    
    #Verifico que se haya invertido en el tiempo
    plt.figure()
    plt.title('ECG filtrado con filtro de mediana')
    plt.plot(x)
    plt.grid() 
    plt.show()    
    
    #Mi template sera qrs_pattern1
    template = qrs_pattern1[:,0]
    template = np.array(template,dtype = float)
    
    #Los filtros del coeficiente FIR se obtienen invirtiendo en el tiempo
    #el template
    fir_coeffs = template[::-1]

    #Aplico el filtro
    det = sig.lfilter(fir_coeffs,1.0,x)
    det = np.power(det,2)
    det = det/np.max(det)
    
    #Grafico la senal de ECG con sus picos R detectados
    plt.figure()
    plt.plot(det,label = 'Detector')
    plt.grid()
    axes_hdl = plt.gca()
    axes_hdl.legend() 
    plt.show()    

    return 

#Script

testbench()