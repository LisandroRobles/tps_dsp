
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 00:00:18 2019

@author: lisandro

En este script se evalua el metodo de segmentacion sobre un canal de un registro ecg de qtdb. 
Se busca reducir el efecto de borde en los segmentos (es decir,que un latido quede cortado a la mitad entre segmentos).
Las variables de interes son:
    
    registro: int
    
        Numero de registro (0-81)
        
    ch: int
    
        Canal del registro (0-1)
    
    window: str
        
        Tipo de ventana aplicada a cada segmento
    
    distance: float 

        Distancia maxima de la anotacion QRS respecto al punto central del 
        segmento a partir de la cual se considera que ha ocurrido un latido
        
    root_dir: str
    
        Directorio en el que se almacenaran los resultados

    nsect: int
    
        Cantidad de segmentos que se desea almacenar en resultados
        
"""

# %% Inclusion de paquetes

#Paquete numerico
import numpy as np

#Paquete para plotear
import matplotlib.pyplot as plt

#Paquete para manejo de archivos de qtDB
import wfdb

#Paquetes para manejo de archivos en general
import glob 

#Paquete de manejo de directorio
import os

#Modulo propio para procesamiento de ecg
import pdsmodulos.ecg as ecg

# %% Testbench

def testbench():
    
    # %% Preparacion del entorno
    
    plt.close('all') #Cierra las ventanas abiertas

    # %% Parametros de registro
    
    registro = 2 #Registro deseado

    ch = 0 #Canal deseado

    # %% Parametros de muestreo
    
    fs = 250 #Hz
    
    # %% Parametros de segmentacion
    
    overlap = 90 #Solapamiento entre segmentos

    before = 0.1 #Largo de la ventana antes del punto (segundos)
    
    after = 0.1 #Largo de la ventana despues del punto (segundos)    

    window = 'hann' #Ventana aplicada a cada segmento

    # %% Parametros de etiquetado
    
    distance = 0.06 #Distancia de anotacion respecto al punto
    
    # %% Parametros de almacenamiento
    
    root_dir = '/home/lisandro/dsp/tps_dsp/proyecto/segmentos/'  #Directorio

    win_dir = window + '/' #Carpeta de ventana

    dist_dir = str(distance) + '/' #Carpeta de distancia

    reg_dir = str(registro) + '/' #Carpeta de registro

    ch_dir = str(ch) + '/'  #Carpeta de canal

    categoria = ['NO/','SI/'] #Categorias

    nsect = 2000 #Cantidad de segmentos que se almacenan en resultados

    # %% Lectura de registros
    
    paths = glob.glob('/usr/qtdb/*.atr') #Levanta el nombre de los registros
    
    paths = [path[:-4] for path in paths] #Les quita la extension
    
    paths.sort() #Los ordena por orden alfabetico

    # %% Lectura del canal 

    #Lee todos los canales del registro actual
    signals, fields = wfdb.rdsamp(paths[registro])

    #Lee las anotaciones del registro actual
    annotation = wfdb.rdann(paths[registro],'atr')   
    peaks = annotation.sample

    #Obtiene el canal deseado
    x = signals[:,ch]

    # %% Filtrado

    #Filtrado con dwt (nivel isoelectrico nulo)
    x = ecg.dwt_filter(x,wav = 'db4',levels = 6,level_min = 1,level_max = 6)

    # %% Normalizacion

    x = ecg.normalizar(x) #Normaliza los latidos

    # %% Segmentacion

    #Divide el ecg en segmentos
    x_sect,points = ecg.segmentar(x,before = before,after = after,overlap = overlap,fs = fs,window = window)
    
    # %% Etiquetados
    
    #Etiqueta cada segmento
    labels = ecg.etiquetar(x_sect,points,peaks,fs = fs,distance = distance)
    
    # %% Almacenamiento de resultados

    #Crea el directorio
    for i in range(np.size(categoria)):
        try:
            os.makedirs(root_dir + dist_dir + win_dir + reg_dir + ch_dir + str(categoria[i]))
        except:
            pass
        
    fig = plt.figure()
    maximo = x_sect.max()
    minimo = x_sect.min()
    for i in range(nsect):
        plt.clf()
        plt.cla()
        plt.axis('off')
        plt.plot(x_sect[i,:])
        plt.ylim(bottom = minimo,top = maximo)
        f_name = root_dir + dist_dir + win_dir + reg_dir + ch_dir + str(categoria[labels[i]]) + str(points[i]) + '.png'
        plt.savefig(f_name)
    
    
# %% Ejecuta el testbench

testbench()