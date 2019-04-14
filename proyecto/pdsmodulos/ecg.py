#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 16:51:01 2018

@author: lisandro

Este modulo implementa funciones utiles para el procesamiento de los registros
de ECG.

Se importa de la siguiente manera:
    
    

"""


import numpy as np
import scipy.signal.windows as win
import pywt


def dwt_filter(x,wav = 'db4',levels = 6,level_min = 2,level_max = 3):
    '''
    
    Filtra una señal de ecg aplicando analisis multinivel con la dwt y quedandose 
    solo con los coeficientes entre level_min y level_max
    
    Parameters
    ----------
    x: numpy.ndarray ndim = 1
        
        Señal de ECG.
    
    wav: str 
    
        Wavelet madre
    
    levels: int
    
        Cantidad de niveles en los que se hace la descomposicion
    
    level_min: int
        
        Nivel minimo en el que se hace la reconstruccion
    
    level_max: int
        
        Nivel maximo en el que se hace la reconstruccion

    Returns
    -------
    x: numpy.ndarray ndim = 1
    
        Señal de ECG reconstruida        
    
    '''

    #Genera la wavelet madre    
    w = pywt.Wavelet(wav)
    
    #Obtiene los coeficientes de la descomposicion
    coeffs = pywt.wavedec(x, w, level=levels)
    
    #Descarta los coeficientes que no estan en el rango deseado
    for i in range(len(coeffs)):
        if (i < level_min) | (i > level_max):
            coeffs[i] = np.zeros_like(coeffs[i])    
    
    #Reconstruye la señal a partir de los coeficientes deseados 
    x = pywt.waverec(coeffs,w)
    
    return x

def segmentar(x,before = 0.1,after = 0.1,overlap = 50,fs = 250,window = 'rectangular'):
    '''
    
    Segmenta un registro de ecg en un conjunto de ventanas. Cada una de las ventanas
    representa el entorno de un punto central dentro de la misma
    
    Parameters
    ----------
    x: numpy.ndarray ndim = 1
        
        Señal de ECG.
    
    before: float
    
        Tiempo de la ventana antes del punto central [seg.]
    
    after: float
    
        Tiempo de la ventana despues del punto central [seg.]
    
    overlap: float
        
        Solapamiento entre ventanas consecutivas
    
    fs: int
        
        Frecuencia de muestreo [Hz]

    window: str
    
        Ventana que se aplicara a cada segmento 

    Returns
    -------
    x_sect: numpy.ndarray ndim = 2
    
        Matriz en donde cada fila corresponde a un segmento
        
    points: numpy.ndarray ndim = 1
    
        Punto central de cada segmento
    
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
    
    #Vector donde se indica el punto que representa cada segmento
    points = np.zeros((nsect,),dtype = int)
    
    #Para cada segmento
    for i in range(nsect):

        #Obtengo el bloque i-esimo de todas las realizaciones
        xi = x[int(n1):int(n1 + L)]
        
        #Ventanea
        xwi = xi*w
        
        #Almacena el segmento actual en la matriz
        x_sect[i,:] = xwi
        
        #Obtiene el punto correspondiente al segmentos
        points[i] = n1 + before
        
        #Muevo el contador al inicio del proximo segmento
        n1 = n1 + n0

    return x_sect,points

def etiquetar(x_sect,points,peaks,fs = 250,distance = 0.1):
    '''
    
    Recibe los segmentos y los puntos centrales que corresponden a cada segmento 
    y en base a las anotaciones QRS y a la variable distance determina si en 
    ese segmento ha ocurrido un latido 

    Parameters
    ----------
    x_sect: numpy.ndarray ndim = 2
        
        Matriz en donde cada fila corresponde a un segmento de un ecg original
    
    points: numpy.ndarray ndim = 1 
    
        Punto central de cada segmento
    
    fs: int
    
        Frecuencia de muestreo [Hz]
    
    peaks: numpy.ndarray ndim = 1
        
        Anotaciones QRS del registro ecg
    
    distance: float
        
        Distancia maxima de la anotacion QRS respecto al punto central del 
        segmento a partir de la cual se considera que ha ocurrido un latido

    Returns
    -------
    labels: numpy.ndarray ndim = 1
    
        Vector binario que indica si ha ocurrido o no un latido en el segmento
        correspondiente
    
    '''
    
    #Pasa la distancia de tiempo a muestras
    distance = int(distance*fs)
    
    #Cantidad de segmentos
    nsect = np.size(points)

    #Vector donde se indicara si en ese segmento ocurre un latido
    labels = np.zeros((nsect,),dtype = bool)    

    #Contador de inicio de cada segmento
    n1 = 0

    #Paso entre segmentos
    n0 = points[1] - points[0]
    
    #Largo de cada segmento
    L = np.size(x_sect,1)
    
    #Para cada segmento
    for i in range(nsect):

        #Determina si en el segmento actual hay un latido
        #El criterio es si hay una anotación en el segmento actual
        condition = ( peaks > n1 ) & ( peaks <= (n1 + L) )
        
        #Si la anotación ocurre en el segmento actual
        if np.size(np.where(condition == True)) != 0:
            
            #Obtiene el indice de la anotacion
            index = peaks[np.where(condition == True)]
            
            #Finalmente, asigna que ha ocurrido un latido si la anotacion
            #ha ocurrido en el rango de +- distance [seg] respecto al punto
            #que representa el segmento
            if (index > (points[i] - distance)) & (index <= (points[i] + distance)):   
                labels[i] = True

        #Muevo el contador al inicio del proximo segmento
        n1 = n1 + n0

    return labels        

def normalizar(x,before = 0.24,after = 0.24,fs = 250):
    '''
    
    Normaliza los latidos de un ecg generando una funcion de normalizacion. Para
    esto se toma una ventana correspondiente a cada punto y se asigna a la funcion
    de normalizacion el valor de maxima magnitud de la ventana.
    
    Parameters
    ----------
    x: numpy.ndarray ndim = 1
        
        Señal de ECG.
        
    before: float
    
        Tiempo anterior al punto central de la ventana [seg.]
    
    after: float
        
        Tiempo posterior al punto central de la ventana [seg.]
    
    fs: int
        
        Frecuencia de muestreo [Hz]

    Returns
    -------
    x: numpy.ndarray ndim = 1
    
        Señal de ECG normalizada      
    
    '''
        
    #Pasa los parametros a muestras
    before = int(before*fs)
    after = int(after*fs)
    
    #Genera g donde se almacenara el ecg normalizado
    N = np.size(x)
    g = np.zeros((N,),dtype = float)
    
    #Zero padea x
    xp = np.pad(x,((before,after)),'constant')    
    
    for i in range(N):
        window = xp[i:int(i + after + before + 1)]
        maximo = np.max(window)
        minimo = np.min(window)
        g[i] = max(abs(maximo),abs(minimo))
    
    g[np.where(g < 0.5*np.mean(g))] = 0.5*np.mean(g)
    g = np.convolve(g,np.ones((120,)),mode = 'same')/120
    g[np.where(g < 0.5*np.mean(g))] = 0.5*np.mean(g)

    x = x/g
    
    return x