#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:44:05 2019

@author: lisandro

En este script se carga una arquitectura de red neuronal ya entrenada y se la 
aplica a un ECG de la qtDB en particular para detectar los complejos QRS y com
pararla con las anotaciones.
Luego, se usa ambas detecciones para estimar la frecuencia cardiaca.
"""
# %% Inclusion de paquetes

import numpy as np # Paquete numerico

import matplotlib.pyplot as plt # Paquete para graficar

import glob # Paquetes para manejo de archivos en general

import wfdb # Paquete para manejo de archivos de qtDB

import pdsmodulos.ecg as ecg #Modulo propio para procesamiento de ecg

import pywt # Paquete de wavelets

from keras import models #Para importar el modelo

from keras.preprocessing.image import load_img

## %% Funciones

def estandarizar(x):
    
    x = x - np.mean(x)
    
    x = x/np.std(x)
    
    return x

def escalograma(x,min_scale = 1,\
                max_scale = 31,step = 1,wavelet = 'gaus2',Ts = 0.004):
    
    scales = np.arange(min_scale,max_scale,step) # Vector de escalas
    
    # APlica la cwt con los parametros seleccionados
    
    cwtmatr,freqs = pywt.cwt(x,\
                         scales,\
                         wavelet,\
                         sampling_period = Ts)
    
    sc = np.power(cwtmatr,2) # Obtiene el escalograma de potencia 
    
    del cwtmatr # Libera el espacio de cwtmatr
    
    for k in range(len(scales)): # Se normaliza en escala (cwt es sesgado)

        sc[k,:] = np.divide(sc[k,:],scales[k])
        
    return sc

def generar_imagen(sc,fig,min_scale = 1,max_scale = 31,\
                   vmin = 0,vmax = 60.7563):

        image_height = int(np.size(sc,0))
            
        image_width = int(np.size(sc,1))
        
        my_dpi = 1
        
        plt.clf()
        
        plt.cla()
        
        plt.axis('off')
        
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        
        ax.set_axis_off()
        
        fig.add_axes(ax)
        
        plt.imshow(sc,extent = [0,image_width,max_scale,min_scale]\
                                ,cmap = 'plasma',interpolation = 'nearest',\
                                aspect = 'auto',vmin = vmin,vmax = vmax)              
                
        f_name = 'temp.png'
        
        plt.savefig(f_name,dpi = my_dpi)
        
        img = load_img(f_name,target_size=(image_height,image_width))
        
        return img

def predict_CNN(img,model):
    
    image_height = int(np.size(img,0))    

    image_width = int(np.size(img,1))    

    channels = 3    

    img = np.reshape(img,[1,image_height,image_width,channels])

    prediction = model.predict(img)[:,1]
        
    return prediction

def QRSdetection_CWT_CNN(x,model,\
                         before = 0.1,after = 0.1,overlap = 0,fs = 250,\
                         min_scale = 1,max_scale = 31,step = 1,wav = 'gaus2',\
                         vmin = 0,vmax = 60.7563):
        
    Ts = 1/fs     # Periodo de muestreo
        
    before = int(before*fs) # Pasa los parametros a muestras
    
    after = int(after*fs)
    
    n = np.size(x,0) # Largo de la señal
    
    L = before + after + 1 # Largo del segmento
    
    n1 = 0  # Contador para mover el inicio de cada segnemto    
    
    n0 = int(np.floor((1-(overlap/100))*int(L))) # Pasos entre segmentos
    
    nsect = int(1 + np.floor((n-L)/(n0))) # Cantidad de segmentos

    qrs_detection = [] # Lista donde se almacenan los indices

    prediccion_anterior = 0 # Prediccion anterior

    n_scales = int((max_scale - min_scale)/(step))

    my_dpi = 1

    # Fig en la que se iran generando las imagenes

    fig = plt.figure(figsize=(L/my_dpi,\
                              n_scales/my_dpi), dpi=my_dpi)
    
    x = estandarizar(x) # Estandariza x (u = 0 y v = 1)

    # Para cada segmento
    
    for i in range(nsect):
                
        xi = x[int(n1):int(n1 + L)]  # Obtiene el segmento i-esimo
    
        # Obtiene el escalograma de potencia del segmento
    
        sc = escalograma(xi,min_scale = min_scale,\
                 max_scale = max_scale,step = step,\
                 wavelet = wav,Ts = Ts)

        sc = np.divide(sc,np.var(x)) # Normaliza respecto a la energia de x

        img = generar_imagen(sc,fig,vmin = vmin,vmax = vmax) # Genera imagen
        
        prediccion = predict_CNN(img,model) # Usa la red y la imagen para pred.
        
        if (prediccion == 1):
            
            if (prediccion_anterior == 0):
                
                qrs_detection.append(int(n1 + before))
                
            else:
                
                qrs_detection[-1] = int(n1)
        
            prediccion_anterior = 1
            
        else:
            
            prediccion_anterior = 0 
        
        # Muevo el contador al inicio del proximo segmento
        n1 = n1 + n0

    # Paso la lista de indices a numpy array
    
    qrs_detection = np.array(qrs_detection)
    
    return qrs_detection

# %% Testbench

def testbench():
    
    # %% Preparacion del entorno
        
    plt.close('all') #Cierra las ventanas abiertas

    # %% Parametros de registro
    
    registro = 49 # Numero de registro [0:81]

    # %% Parametros de muestreo
    
    fs = 250 # Frecuencia de muestreo [Hz]
    
    # %% Parametros de segmentacion
    
    overlap = 0 #Solapamiento entre segmentos

    before = 0.1 #Largo de la ventana antes del punto (segundos)
    
    after = 0.1 #Largo de la ventana despues del punto (segundos)    

    # %% Parametros de la red
    
    model_name = 'resultados/cnn4.h5'

    # %% Parametros de la CWT

    min_scale = 1 # Escala minima
    
    max_scale = 31 # Escala maxima
    
    step = 1 # Paso entre escalas
                
    vmax = 60.7563 # Normalizacion maxima
    
    vmin = 0 # Normalizacion minima
    
    # Wavelet madre
    #'cgau1', 'cgau2', 'cgau3','cgau4','cgau5','cgau6','cgau7','cgau8',
    #'cmor',
    #'fbsp',
    #'gaus1','gaus2','gaus3','gaus4','gaus5','gaus6','gaus7','gaus8',
    #'mexh',
    #'morl',
    #'shan'
    wavelet = 'gaus2'

    # %% Lectura de archivos

    paths = glob.glob('/usr/qtdb/*.atr') #Levanta el nombre de los registros
    
    paths = [path[:-4] for path in paths] #Les quita la extension
    
    paths.sort() #Los ordena por orden alfabetico

    # %% Lectura de registro
    
    signals, fields = wfdb.rdsamp(paths[registro]) # Lee las derivaciones 
    
    annotation = wfdb.rdann(paths[registro],'atr') # Lee las anotaciones   
    qrs_detection_standard = annotation.sample

    x = signals[:,0] # Se queda con la primer derivacion

    # %% Generacion de las imagenes    

    model = models.load_model(model_name)

    qrs_detection = QRSdetection_CWT_CNN(x,model,\
                                         before = before,after = after,\
                                         overlap = overlap,fs = fs,\
                                         min_scale = min_scale,\
                                         max_scale = max_scale, step = step,\
                                         wav = wavelet,vmin = vmin,vmax = vmax)

    return qrs_detection_standard,qrs_detection
    
# %% Ejecuta el testbench

qrs_detection_standard,qrs_detection = testbench()