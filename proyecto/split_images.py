#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:47:47 2019

@author: lisandro

En este script se toma un directorio de imagenes clasificadas en carpetas de
distintas clases y se las divide en conjuntos de entrenamiento, validacion y 
prueba

Ejemplo:
    
    imagenes/SI : 40 imagenes
    imagenes/NO : 60 imagenes

    test_split = 0.2
    validation_split = 0.1
    
    Produce los siguientes directorios
    
    imagenes/train/SI  
    imagenes/train/NO 
    imagenes/valid/SI 
    imagenes/valid/NO 
    imagenes/test/SI 
    imagenes/test/NO 

    Con la proporcion indicada

"""
# %% Inclusion de paquetes

import os #

import numpy as np #Paquete numerico

import shutil #

# %% Testbench

def testbench():

    # %% Parametros del dataset
    
    #Directorio de las imagenes
    
    root_dir = '/home/lisandro/dsp/tps_dsp/proyecto/escalogramas'
    
    classes = ['/SI','/NO'] #Clases en las que estan clasificadas las imagenes
    
    # %% Parametros de la division
    
    validation_split = 0.1 #Procentaje de imagenes para validacion
    
    test_split = 0.1 #Porcentaje de imagenes para prueba
    
    # %% Division del dataset
    
    for class_ in classes: #Para cada clase
    
        os.makedirs(root_dir +'/train' + class_) #Crea carpeta de entrenamiento
        
        os.makedirs(root_dir +'/valid' + class_) #Crea carpeta de validacion
        
        os.makedirs(root_dir +'/test' + class_) #Crea carpeta de prueba
        
        currentCls = class_ #Clase actual
        
        src = root_dir + currentCls # Carpeta de donde se copian las imagenes
        
        allFileNames = os.listdir(src) #Obtiene todas las imagenes de una clase
        
        np.random.shuffle(allFileNames) #Mezcla al azar las imagenes 
        
        #Divide las imagenes 
        
        train_FileNames\
        ,val_FileNames\
        ,test_FileNames = np.split(np.array(allFileNames)\
                                   ,[int(len(allFileNames)\
                                         *(1-(test_split + validation_split)))\
                                     , int(len(allFileNames)\
                                           *(1-(test_split)))])

        # Ubicacion de las imagenes de entrenamiento
        
        train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
        
        # Ubicacion de las imagenes de validacion
        
        val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
        
        # Ubicacion de las imagenes de prueba
        
        test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
        
        # Imprime la cantidad total de imagenes
        
        print('Total images: ', len(allFileNames))
        
        # Imprime la cantidad de imagenes de entrenamiento
        
        print('Training: ', len(train_FileNames))
        
        # Imprime la cantidad de imagenes de validacion
        
        print('Validation: ', len(val_FileNames))
        
        # Imprime la cantidad de imagenes de prueba
        
        print('Testing: ', len(test_FileNames))
        
        for name in train_FileNames: # Para cada imagen de entrenamiento

            shutil.copy(name,root_dir + "/train"+currentCls) # Copia la imagen
        
        for name in val_FileNames: # Para cada imagen de validacion
            
            shutil.copy(name,root_dir + "/valid"+currentCls) # Copia la imagen
        
        for name in test_FileNames: # Para cada imagen de prueba
            
            shutil.copy(name,root_dir + "/test"+currentCls) # Copia la imagen
            
# %% Ejecucion del testbench
            
testbench()