#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 16:51:01 2018

@author: lisandro
"""
import numpy as np
import scipy.signal.windows as win

def segmentar(x,before = 0.1,after = 0.1,overlap = 50,fs = 250,window = 'rectangular'):
    '''
    Segmenta un registro de ecg en ventanas que representan a un punto del ecg

    <x> Señal de ECG
    <before> Tiempo en segundos antes del punto que representa al segmento
    <after> Tiempo en segundos después del punto que representa al segmento
    <overlap> Solapamiento entre ventanas consecutivas
    <fs> Frecuencia de muestreo de
    <window> Ventana que se aplicara a cada segmento
    '''
    
    #Pasa los parametros a muestras
    before = int(before*fs)
    after = int(after*fs)
    
    #Largo de la señal
    n = np.size(x,0)
    
    #Largo del segmento
    L = before + after + 1
    
    #Contador para mover el inicio de cada segnemto    
    n1 = 0
    
    #Pasos entre segmentos
    n0 = int(np.floor((1-(overlap/100))*int(L)))
    
    #Cantidad de segmentos
    nsect = int(1 + np.floor((n-L)/(n0)))
            
    #Genera la ventana que se aplicara en cada segmento
    if window == 'rectangular':
        w = win.boxcar(L)
    elif window == 'bartlett':
        w = win.bartlett(L)
    elif window == 'hann':
        w = win.hann(L)
    elif window == 'hanning':
        w = win.hanning(L)
    elif window == 'hamming':
        w = win.hamming(L)
    elif window == 'flattop':
        w = win.flattop(L)
    else:
        w = win.boxcar(L)
    
    #Matriz donde se almacenaran todos los segmentos
    x_sect = np.zeros((nsect,L),dtype = float)
    
#    #Vector donde se indicara si en ese segmento ocurre un latido
#    labels = np.zeros((nsect,),dtype = bool)    
#    
#    #Vector donde se indica el punto que representa cada segmento
#    points = np.zeros((nsect,),dtype = int)
    
    #Para cada segmento
    for i in range(nsect):

        #Obtengo el bloque i-esimo de todas las realizaciones
        xi = x[int(n1):int(n1 + L)]
        
        #Ventanea
        xwi = xi*w
        
        #Almacena el segmento actual en la matriz
        x_sect[i,:] = xwi
        
#        #Obtiene el punto correspondiente al segmentos
#        points[i] = n1 + before
#        
#        #Determina si en el segmento actual hay un latido
#        #El criterio es si hay una anotación en el segmento actual
#        condition = ( peaks > n1 ) & ( peaks <= (n1 + L) )
#        
#        #Si la anotación ocurre en el segmento actual
#        if np.size(np.where(condition == True)) != 0:
#            
#            #Obtiene el indice de la anotacion
#            index = peaks[np.where(condition == True)]
#            
#            #Finalmente, asigna que ha ocurrido un latido si la anotacion
#            #ha ocurrido en el rango de +- distance [seg] respecto al punto
#            #que representa el segmento
#            if (index > (points[i] - distance)) & (index <= (points[i] + distance)):   
#                labels[i] = True
        
        #Muevo el contador al inicio del proximo segmento
        n1 = n1 + n0

    return x_sect

def etiquetar(x_sect,peaks,distance = 0.05):
    
    

def normalizar(x,before = 0.24,after = 0.24,fs = 250):
    '''
    Normaliza los latidos de un ecg generando una funcion de normalizacion basado
    en el max(abs(max),abs(min)) de cada ventana

    <x> Señal de ECG
    <before> Tiempo en segundos antes del punto que representa al segmento
    <after> Tiempo en segundos después del punto que representa al segmento
    <fs> Frecuencia de muestreo 
    '''
    
    
    #Pasa los parametros a muestras
    before = int(before*fs)
    after = int(after*fs)
    
    #Genera g donde se almacenara el ecg normalizado
    N = np.size(x)
    g = np.zeros((N,),dtype = float)
    
    #Zero padea x
    x = np.pad(x,((before,after)),'constant')    
    
    for i in range(N):
        window = x[i:int(i + after + before + 1)]
        maximo = np.max(window)
        minimo = np.min(window)
        g[i] = max(abs(maximo),abs(minimo))
    
    g[np.where(g < 0.5*np.mean(g))] = 0.5*np.mean(g)
    g = np.convolve(g,np.ones((120,)),mode = 'same')/120
    g[np.where(g < 0.5*np.mean(g))] = 0.5*np.mean(g)

    x = x/g
    
    return x