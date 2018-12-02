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
from scipy.interpolate import spline

import pdsmodulos.spectrum_analyzer as sa
import statistics as st

#Paquete de manejo de datos
import pandas as pd

#Medicion de tiempo
import time 


#Funciones

def filtro_mediana(x,neigh = 0):
        
    N = np.size(x,0)
    S = np.size(x,1)
    
    n = int(2*neigh) + 1
    
    if n > N:
        
        if (N % 2) == 0:
            neigh = int((N - 2)/2)
        else:
            neigh = int((N-1)/2)
    
    
    ymed = np.zeros((N,S),dtype = float)
    
    centro_grilla = neigh 
    
    for i in range(0,N):

        if ( ((i - neigh) > 0) and ((i + neigh) < N) ):
            centro_grilla = i
        
        if neigh is 0:
            r = x[int(centro_grilla),:]
        else:
            r = x[int(centro_grilla - neigh):int(centro_grilla + neigh + 1),:]
            
            
        ymed[i,:] = np.median(r)
                    
    return ymed
        
def decimate(x,fs1,fs2):
    
    #Antes de submuestrearhay que aplicar un filtro anti-allias
    #con frecuencia de corte en la nueva frecuencia de Nyquist
    fny1 = fs1/2
    fny2 = fs2/2
    
    fp = fny2/fny1
    fs = (fny2 + 1)/fny1 
    Ripple = 0.5 #dB
    Att = 50 #dB
    
    lpf_sos_butter = sig.iirdesign(wp = fp, ws = fs, gpass = Ripple, gstop = Att, analog=False, ftype='butter', output='sos')
    
    xfilt = sig.sosfiltfilt(lpf_sos_butter,x,axis = 0)
    
    #Una vez muestrado procedo a diezmar a la senial
    muestras_salteadas = int(fs1/fs2)
    N = int(np.size(x,axis = 0))
    
    #Senial decimada
    xd = xfilt[0:N:muestras_salteadas,:]
    
    #Vector de tiempo para la senial decimada
    Nd = np.size(xd,0)
    Ts2 = 1/fs2
    td = np.linspace(0,Ts2*(Nd-1),Nd)
    
    return (td,xd)
    

        

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
    fs1 = 1000
    N = 20000
    fs2 = 20

    #Almaceno los resultados
    tus_resultados = []

    #Primero estimo el movimiento de linea SIN DIEZMAR
    
    start = time.time()
    
    ecg_lead_med200 = sig.medfilt(ecg_lead,[int(0.2*fs1)+1,1])
    ecg_lead_med600_sin_diezmar = sig.medfilt(ecg_lead_med200,[int(0.6*fs1)+1,1])    
    B_sin_diezmado = ecg_lead_med600_sin_diezmar
    
    end = time.time()
    time_sin_diezmar = np.around(end - start,decimals = 3)
    
    #Ahora CON DIEZMADO
    
    start = time.time()
    
    (t,ecg_lead_diezmado) = decimate(ecg_lead,fs1,fs2)
    ecg_lead_med200 = sig.medfilt(ecg_lead_diezmado,[int(0.2*fs2)+1,1])
    ecg_lead_med600_con_diezmado = sig.medfilt(ecg_lead_med200,[int(0.6*fs2)+1,1])
    B_con_diezmado = np.zeros((np.size(ecg_lead,0),1),dtype = float)
    B_con_diezmado[0:np.size(ecg_lead,0):int(fs1/fs2)] = ecg_lead_med600_con_diezmado
    
    end = time.time()
    time_diezmando = np.around(end-start,decimals = 3)
    
    tus_resultados.append([str(time_sin_diezmar),str(time_diezmando)])
    
    df = pd.DataFrame(tus_resultados, columns = ['tiempo filtro de mediana sin diezmado (s)','tiempo filtro de mediana con diezmado (s)'])
    
    print(df)
    
    plt.figure()
    plt.title('Estimacion de linea de base')
    plt.plot(ecg_lead,label = 'Sin filtrar')
    plt.plot(B_sin_diezmado,label = 'B sin diezmado')
    plt.plot(B_con_diezmado,label = 'B con diezmado')
    #plt.plot(np.power((B_sin_diezmado - B_con_diezmado),2),label = 'Error cuadratico')
    plt.grid()
    axes_hdl = plt.gca()
    axes_hdl.legend()  
    plt.show()

    plt.figure()
    plt.title('Error cuadratico de diezmar')
    plt.plot(np.power((B_con_diezmado - B_sin_diezmado),2),label = 'B sin diezmado')
    plt.grid()
    axes_hdl = plt.gca()
    axes_hdl.legend()  
    plt.show()
    
    x_sin_diezmado = ecg_lead - B_sin_diezmado
    x_con_diezmado = ecg_lead - B_con_diezmado

    plt.figure()
    plt.title('Estimacion de activaidad electrica')
    plt.plot(x_sin_diezmado,label = 'Actividad electrica sin diezmado')
    plt.plot(x_con_diezmado,label = 'Actividad electrica con diezmado')
    plt.grid()
    axes_hdl = plt.gca()
    axes_hdl.legend()  
    plt.show()
    
#Script

testbench()