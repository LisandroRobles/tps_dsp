#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:22:47 2019

@author: lisandro

En este script se prueba fine tuning de un modelo preentrenado de la libreria
keras.
    
"""
# %% Inclusion de paquetes

import numpy as np #Paquete numerico

import matplotlib.pyplot as plt #Paquete para graficar

import itertools #Necesario para graficar la matriz de confusion

from keras.applications import vgg16 #Importa modelo preentrenado

from keras import models #Para crear un modelo

from keras import layers #Para agregar capas

from keras import optimizers #Para seleccionar el optimizador de la red

from keras import regularizers #Regularizacion -> Evitar overfitting

from keras.preprocessing.image import ImageDataGenerator #Manejar las imagenes

from sklearn.metrics import confusion_matrix #Para calcular la confusion matrix

# %% Definicion de funciones

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %% Testbench

def testbench():
    
    # %% Preparacion del entorno
        
    plt.close('all') #Cierra las ventanas abiertas
    
    # %% Parametros del dataset
    
    # Carpeta de entrenamiento
    
    train_path = '/home/lisandro/dsp/tps_dsp/proyecto/escalogramas/train/'
    
    # Carpeta de validacion
    
    valid_path = '/home/lisandro/dsp/tps_dsp/proyecto/escalogramas/valid/'
    
    # Carpeta de test
    
    test_path = '/home/lisandro/dsp/tps_dsp/proyecto/escalogramas/test/'
    
    # Formato de las imagenes
    
    image_size = 50 #Pixels de la imagen NxN
    
    channels = 3 #Cantidad de canales (color o escala de grises)
    
    # Categorias en las que se encuentran clasificadas las imagenes 
    
    labels = ['NO','SI']
    
    # %% Parametros del entrenamiento
    
    # Change the batchsize according to your system RAM
    
    train_batchsize = 100 #Tamaño del batch de entrenamiento
    
    val_batchsize = 10 #Tamaño del batch de validación
    
    epochs = 4 #Epocas (cantidad de veces que se itera)    
    
    # %% Carga la red preentrenada
    
    #Carga el modelo preentrenado
    #VGG16
    #INCLUDE_TOP: Si se incluye o no una capa densa al final
    #WEIGHTS: Carga para la arquitectura vgg16 los pesos entrenados con la 
    #db de imagenet
    #INPUT_SHAPE: Tamaño de la imagen
    
    base_model = vgg16.VGG16(include_top=False\
                             , weights='imagenet'\
                             ,input_shape = (image_size,image_size,channels))
    
    #Imprime informacion de la red preentrenada
    
    base_model.summary()
    
    # %% Creacion del modelo nuevo
    
    # Se entrenaran solo las ultimas 4 capas de la red preentrenada
    
    for layer in base_model.layers[:-4]:
        
        layer.trainable = False
 
    # Verifica las capas entrenables de la red preentrenada
    
    for layer in base_model.layers:
        
        print(layer, layer.trainable)
    
    # Modelo en que se almacenara la nueva red
    
    model = models.Sequential()
     
    #Agrega a la nueva red la red preentrenada
    
    model.add(base_model)
     
    # Agrega una capa de flatten
    
    model.add(layers.Flatten())
    
    # Agrega una capa densa
    
    model.add(layers.Dense(1024\
                           ,activation='relu'\
                           ,kernel_regularizer = regularizers.l2(0)))
    
    #Agrega una capa de dropout -> Evitar overfitting
    
    model.add(layers.Dropout(0.5))
    
    #Agrega una capa densa
    
    model.add(layers.Dense(2, activation='softmax'))
     
    #Imprime informacion de la nueva red
    
    model.summary()

    # Optimizador de la red
    
    optimizer=optimizers.RMSprop(lr=1e-4)
    #sgd = optimizers.SGD(lr=1e-4)
    
    #Compila la red
    
    model.compile(loss='categorical_crossentropy'\
                  ,optimizer=optimizer\
                  ,metrics=['acc'])
    
    # %% Preparacion de los datos
    
    #Genera el batch de entrenamiento    
    
    train_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(train_path\
                                      ,target_size=(image_size,image_size)\
                                      ,classes=labels\
                                      ,batch_size=train_batchsize)
    
    #Genera el batch de validacion
    
    valid_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(valid_path\
                                      ,target_size=(image_size,image_size)\
                                      ,classes=labels,\
                                      batch_size=val_batchsize)
    
    #Genera el batch de prueba
    
    test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path\
                                     ,target_size=(image_size,image_size)\
                                     ,classes=labels)
        
    # %% Entrenamiento de la red
    
    # Entrena la red
    
    model.fit_generator(train_batches\
                        ,steps_per_epoch=\
                            train_batches.samples/train_batches.batch_size\
                        ,validation_data = valid_batches\
                        ,validation_steps = \
                            valid_batches.samples/valid_batches.batch_size\
                        ,epochs = epochs\
                        ,verbose = 1)     
    
    # Guarda el modelo entrenado
    
    model.save('detector.h5')
    
    # %% Testeo de la red
    
    #Obtiene las imagenes de prueba y los labels correspondientes
    
    test_imgs,test_labels = next(test_batches)

    #Obtiene los labels del conjunto de prueba

    test_labels = test_labels[:,0]

    #Utiliza la red entrenada para predecir sobre el conjunto de prueba

    predictions = model.predict_generator(test_batches,steps = 1,verbose = 0)
    
    # Como la activacion de la ultima capa es softmax, la salida esta expresada
    #en terminos de probabilidad. En caso que sea mayor a 0.5 se asigna 1 y si
    #es menor se asigna 0
    
    predictions = np.where(predictions > 0.5,1,0)
    
    # Calcula la matriz de confusion usando los labels predichos y los reales
    
    cm = confusion_matrix(test_labels,predictions[:,0])
    
    # Categorias que se imprimen en la matriz de confusion
    
    cm_plot_labels = labels
    
    # Imprime la matriz de confusion
    
    plot_confusion_matrix(cm\
                          ,cm_plot_labels\
                          ,title='confusion matrix'\
                          ,normalize = True)
    
# %% Ejecuta el testbench

testbench()