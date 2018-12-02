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

import pdsmodulos.spectrum_analyzer as sa
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
    ecg_lead = ecg_lead.reshape(np.size(ecg_lead),1)
    N = np.size(ecg_lead,0)
    n = np.linspace(0,N-1,N)
    
    qrs_detections = mat_struct["qrs_detections"]
    qrs_detections = qrs_detections[:,0]
    
    #peak = np.zeros((N,1),dtype = float)
    #peak[qrs_detections,:] = ecg_lead[qrs_detections,:]
    fs = 1000
    T = int(0.2*fs)
    peak = ecg_lead[qrs_detections,:]
    null_detections = qrs_detections - T
    null = ecg_lead[null_detections,:]

    null_cs = CubicSpline(null_detections,null)
    
    plt.figure()
    plt.plot(n,ecg_lead,label = 'ECG')
    #plt.plot(qrs_detections,peak,'o',label = 'picos')
    plt.plot(null_detections,null,'x',label = 'segmento PQ')
    plt.grid()
    axes_hdl = plt.gca()
    axes_hdl.legend() 
    plt.show()

    plt.figure()
    plt.plot(n,ecg_lead,label = 'ECG')
    plt.plot(n,null_cs(n),label = 'Nivel Isoelectrico')
    plt.grid()
    axes_hdl = plt.gca()
    axes_hdl.legend() 
    plt.show()
    
    ecg_filtrado = ecg_lead - null_cs(n)
    
    plt.figure()
    plt.plot(n,ecg_filtrado,label = 'ECG_Filtrado')
    plt.grid()
    axes_hdl = plt.gca()
    axes_hdl.legend() 
    plt.show()    
                
#Script

testbench()