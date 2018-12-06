#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:44:36 2018

@author: lisandro
"""

#####################################################
# Paquetes necesarios para realizar el ejercicio #6 #
#####################################################

#Paquetes estándar

#Paquete numérico
import numpy as np
#Paquete gráfico
import matplotlib.pyplot as plt
import matplotlib as mpl
# Setup inline graphics: Esto lo hacemos para que el tamaño de la salida, 
# sea un poco más adecuada al tamaño del documento
mpl.rcParams['figure.figsize'] = (10,10)
# Esto tiene que ver con cuestiones de presentación de los gráficos,
# NO ES IMPORTANTE
fig_sz_x = 10
fig_sz_y = 10
fig_dpi = 80 # dpi
fig_font_family = 'Ubuntu'
fig_font_size = 16
plt.rcParams.update({'font.size':fig_font_size})
plt.rcParams.update({'font.family':fig_font_family})
#Paquete de manejo de datos
from pandas import DataFrame
#Paquete para mostrar el dataframe
from IPython.display import HTML
#Paquete de procesamiento de señales
import scipy.signal as sig
#Paquete para levantar ECG_TP4
import scipy.io as sio
#Warning
import warnings
warnings.filterwarnings('ignore')
#Paquete para interpolar
from scipy.interpolate import spline
from scipy.interpolate import CubicSpline

import itertools
from sklearn.metrics import confusion_matrix

#Paquetes propios

#Paquete que implementa funciones relacionadas a filtros digitales
import pdsmodulos.filters as filt
#Paquete que implementa la clase analizador de espectro
import pdsmodulos.spectrum_analyzer as sa

#############
# Funciones #
#############

def detector_latidos(x,template,fs,Vumbral,tiempo_ciego):
    
    #Obtengo el tiempo refractario en muestras
    T = tiempo_ciego
    M = int(T*fs)
    
    #Aplico el filtro adaptado
    det = filt.matched_filter(x,template,Vumbral)
    
    #Obtengo los indices donde no es cero
    square = np.zeros((np.size(det,0),1),dtype = float)
    indices = np.where(det[:,0] > 0)
    indices = np.array(indices,dtype = int).T[:,0]
    square[indices,:] = 1
    
    #Arranco desde el primer indice
    #Aplico el periodo refractario de 300ms entre cada deteccion
    indice_actual = indices[0]
    for nuevo_indice in indices[1:]:
        
        if (nuevo_indice - indice_actual) < M:
            det[nuevo_indice,:] = 0
            
        else:
            indice_actual = nuevo_indice

    #Obtengo los nuevos indices
    indices = np.where(det[:,0] > 0)
    indices = np.array(indices,dtype = int).T[:,0] 
    
    return indices,square


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#############
# Testbench #
#############

def testbench():
    
    #####################
    # Cargo las señales #
    #####################
    
    #Levanto el archivo .mat
    mat_struct = sio.loadmat('./ECG_TP4.mat')
    
    #Levanto el registro de electrocardiograma
    ecg_lead = mat_struct["ecg_lead"]
    ecg_lead = ecg_lead.reshape(np.size(ecg_lead),1)
    
    #Levanto el patrón de latido ventricular
    heartbeat_pattern1 = mat_struct["heartbeat_pattern1"]
    heartbeat_pattern1 = heartbeat_pattern1.reshape(np.size(heartbeat_pattern1),1)
    
    #Levanto el patrón de latido normal
    heartbeat_pattern2 = mat_struct["heartbeat_pattern2"]
    heartbeat_pattern2 = heartbeat_pattern2.reshape(np.size(heartbeat_pattern2),1)
    
    #Levanto el patrón de complejo QRS normal
    qrs_pattern1 = mat_struct["qrs_pattern1"]
    qrs_pattern1 = qrs_pattern1.reshape(np.size(qrs_pattern1),1)  
    
    ########################################
    # Presentación gráfica de los patrones #
    ########################################
    
#    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
#    plt.title('Patrón de latidos')
#    plt.plot(heartbeat_pattern1,label = 'Patrón de latido ventricular')
#    plt.plot(heartbeat_pattern2,label = 'Patrón de latido normal')
#    plt.grid()
#    axes_hdl = plt.gca()
#    axes_hdl.legend() 
#    plt.show()
#    
#    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
#    plt.title('Patrón de complejo QRS')
#    plt.plot(qrs_pattern1,label = 'Patrón QRS normal')
#    plt.grid()
#    axes_hdl = plt.gca()
#    axes_hdl.legend() 
#    plt.show()

    ####################################################################
    # Preprocesamiento del registro de ECG. Filtrado por interpolacion #
    ####################################################################
    
    #Levanto los indices en donde se han detectado los patrones QRS normales
    qrs_detections = mat_struct["qrs_detections"]
    qrs_detections = qrs_detections[:,0]
    
    #Se genera un vector con los valores de los picos detectados
    peak = ecg_lead[qrs_detections,:]
    
    #Muestreo el detector a la misma frecuencia que el registro
    picos = np.zeros((np.size(ecg_lead,0),1),dtype = float)
    picos[qrs_detections] = peak
    
    #Frecuencia de muestreo
    fs = 1000
    
    #A partir de la inspección del registro ECG se estimo que el intervalo entre la onda P y Q ocurre
    #200ms antes del pico R
    #Se lo pasa a unidades de muestras
    T = int(0.1*fs)
    
    #Se genera un vector con los indices del registro en el intervalo entre la onda P y Q
    null_detections = qrs_detections - T
    
    #Se genera un vector con los valores del registro en el intervalo entre la onda P yQ
    null = ecg_lead[null_detections,:]
    
    #En los intervalos intermedios se utiliza una interpolación cúbica
    null_cs = CubicSpline(null_detections,null)
    
    #Genero el vector para interpolar
    N = np.size(ecg_lead,0)
    n = np.linspace(0,N-1,N)
    
    #Interpolo
    null_cs = null_cs(n)
    
    #Filtro el ecg
    ecg_lead = ecg_lead - null_cs

    ############################
    # Aplicación del algoritmo #
    ############################
    
    #Utilizo el patrón QRS normal como template
    #template = qrs_pattern1[:,0]
    #Vumbral = 0.3
    
    #Utilizo el patrón latido normal como template
    template = qrs_pattern1[:,0]
    Vumbral = 0.2
    
    fs = 1000
    tiempo_espera = 0.3
    
    #Aplico el filtrado adaptado, devuelve los indices donde detecto los picos
    indices_deteccion,square_signal = detector_latidos(ecg_lead,template,fs,Vumbral,tiempo_espera)

    #####################################
    # Presentacion grafica de resultados#
    #####################################
        
#    N = np.size(ecg_lead,0)
#    lim_inicial = int(0)
#    lim_final = int(10000)
#    n = np.linspace(lim_inicial,lim_final-1,lim_final - lim_inicial)
#    
#    deteccion_rango = indices_deteccion[(indices_deteccion >= lim_inicial)&(indices_deteccion < lim_final)]
#    deteccion_patron_rango = qrs_detections[(qrs_detections >= lim_inicial)&(qrs_detections < lim_final)]
#    
#    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
#    plt.plot(n,ecg_lead[lim_inicial:lim_final],label = 'ECG')
#    plt.plot(deteccion_rango,ecg_lead[deteccion_rango,:],'x',label = 'Deteccion')
#    plt.plot(deteccion_patron_rango,ecg_lead[deteccion_patron_rango,:],'o',label = 'Deteccion patron')
#    plt.title('Registro ECG')
#    plt.xlabel('$n$')
#    plt.axis('tight')
#    plt.grid()
#    plt.show()
#
#    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
#    plt.plot(n,square_signal[lim_inicial:lim_final])
#    plt.title('Señal cuadrada de la detección')
#    plt.xlabel('$n$')
#    plt.axis('tight')
#    plt.grid()
#    plt.show()

    #########################################
    # Medición del rendimiento del detector #
    #########################################

    deteccion_patron = np.zeros((np.size(ecg_lead,0),1),dtype = int)
    deteccion_patron[qrs_detections,:] = 1
    
    deteccion_algoritmo = np.zeros((np.size(ecg_lead,0),1),dtype = int)
    deteccion_algoritmo[indices_deteccion,:] = 1

    ventana_tiempo = 0.2
    ventana_muestras = int(ventana_tiempo*fs)

    N = int(np.size(deteccion_patron,0))

    residuo = (N % ventana_muestras)
    cant_zeropad = ventana_muestras - residuo
    
    deteccion_patron = np.pad(deteccion_patron,((0,cant_zeropad),(0,0)),'constant')
    deteccion_algoritmo = np.pad(deteccion_algoritmo,((0,cant_zeropad),(0,0)),'constant')

    N = int(np.size(deteccion_patron,0))

    cant_subsets = int(N/ventana_muestras)
    
    subsets_list_patron = np.split(deteccion_patron,cant_subsets,axis = 0)
    subsets_patron = np.hstack(subset for subset in subsets_list_patron)
    prediccion_patron = np.sum(subsets_patron,axis = 0)
    prediccion_patron = prediccion_patron.reshape(np.size(prediccion_patron),1)
    
    subsets_list_algoritmo = np.split(deteccion_algoritmo,cant_subsets,axis = 0)
    subsets_algoritmo = np.hstack(subset for subset in subsets_list_algoritmo)
    prediccion_algoritmo = np.sum(subsets_algoritmo,axis = 0)
    prediccion_algoritmo = prediccion_algoritmo.reshape(np.size(prediccion_algoritmo),1)

    clases = ['No Latido','Latido']

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(prediccion_patron,prediccion_algoritmo)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=clases,title='Confusion matrix, without normalization')
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=clases, normalize=True,title='Normalized confusion matrix')
    
    plt.show()

#    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
#    plt.plot(deteccion_patron)
#    plt.title('Deteccion patron SI/NO')
#    plt.xlabel('$n$')
#    plt.axis('tight')
#    plt.grid()
#    plt.show()
#
#    plt.figure(figsize=(fig_sz_x, fig_sz_y), dpi= fig_dpi, facecolor='w', edgecolor='k')
#    plt.plot(deteccion_algoritmo)
#    plt.title('Deteccion algoritmo SI/NO')
#    plt.xlabel('$n$')
#    plt.axis('tight')
#    plt.grid()
#    plt.show()
        
    return prediccion_patron,prediccion_algoritmo

#Script

prediccion_patron,prediccion_algoritmo = testbench()
