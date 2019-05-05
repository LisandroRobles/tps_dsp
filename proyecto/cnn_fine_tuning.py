#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:22:47 2019

@author: lisandro

En este script se prueba fine tuning de un modelo preentrenado de la libreria
keras.Se la entrena especificando los path de entrenamiento, validacion y 
prueba, el formato de la imagen y los hiperparametros de la red 
(learning rate, optimizador) y del entrenamiento (batch,epocas).
Una vez entrenado, se grafica accuracy y loss durante el entrenamiento y la val
idacion.
Luego, se realiza la prueba y se genera la matriz de confusion para calcular
otros valores como la sensibilidad y el valor predictivo positivo. Finalmente,
se almacena el modelo en un archivo cnnXX.h5 y se guardan los resultados en un
archivo cnnXX.txt
    
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

from contextlib import redirect_stdout # Para guardar model.summary en .txt

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
    
    # %% Parametros de las imagenes
    
    image_height = 32 # Alto de la imagen (cantidad de escalas)
    
    image_width = 32  # Ancho de la imagen (muestras en segmento: T*fs)
    
    channels = 3 # Cantidad de canales (color = 3 o escala de grises = 1)
    
    input_shape = (image_height,image_width,channels) # Formato de las imagenes
        
    labels = ['NO','SI'] # Categorias en las que estan clasificadas 

    n = int(len(labels)) # Cantidad de categorias
    
    # %% Hyperparametros
    
    #(Change the batchsize according to your system RAM)
    
    train_batchsize = 128 # Tamaño del batch de entrenamiento
    
    val_batchsize = 128 # Tamaño del batch de validación
    
    test_batchsize = 128 # Tamaño del batch de prueba
    
    epochs = 1 # Cantidad de veces que se itera sobre el dataset    

    lr = 1e-4 # Learning Rate

    optimizer = optimizers.Adam(lr = lr) # Optimizador
    #optimizer=optimizers.RMSprop(lr = lr)
    #optimizer = optimizers.SGD(lr = lr)

    opt = 'Adam' # Variable a los efectos de documentar el optimizador usado
    
    loss = 'binary_crossentropy' # Func. utilizada para optimizar el algoritmo
    
    metrics = 'acc' # Metrica utilizada para medir la performance

    dropout = 0.25

    regularizer = regularizers.l2(0)

    act_fn = 'relu' # Funcion de activacion de todas las capas menos la ultima
    
    act_fn_final = 'softmax' # Funcion de activacion de la ultima capa

    trainable = 4 # Se entrenan las ultimas n capas convolucionales de vgg16

#    monitor = 'val_loss'
#    
#    patience = 20

    file_model = 'vgg16_v1.h5'
            
    file_results = 'vgg16_v1.txt'  

    
    # %% Carga la red preentrenada
    
    #Carga el modelo preentrenado
    #VGG16
    #INCLUDE_TOP: Si se incluye o no una capa densa al final
    #WEIGHTS: Carga para la arquitectura vgg16 los pesos entrenados con la 
    #db de imagenet
    #INPUT_SHAPE: Tamaño de la imagen
    
    base_model = vgg16.VGG16(include_top=False\
                             , weights='imagenet'\
                             ,input_shape = input_shape)
    
    #Imprime informacion de la red preentrenada
    
    base_model.summary()
    
    # %% Creacion del modelo nuevo
    
    # Se entrenaran solo las ultimas 4 capas de la red preentrenada
    
    for layer in base_model.layers[:-int(trainable)]:
        
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
                           ,activation=act_fn\
                           ,kernel_regularizer = regularizer))
    
    #Agrega una capa de dropout -> Evitar overfitting
    
    model.add(layers.Dropout(dropout))
    
    #Agrega una capa densa
    
    model.add(layers.Dense(n, activation=act_fn_final))
     
    #Imprime informacion de la nueva red
    
    model.summary()
    
    #Compila la red
    #Binary_crossentropy
    model.compile(loss='categorical_crossentropy'\
                  ,optimizer=optimizer\
                  ,metrics=['acc'])
    
    # %% Preparacion de los datos
    
    #Genera el batch de entrenamiento    
    
    train_batches = \
    ImageDataGenerator(rescale=1./255).flow_from_directory(train_path\
                                      ,target_size=(image_height,image_width)\
                                      ,classes=labels\
                                      ,batch_size=train_batchsize)
    
    #Genera el batch de validacion
    
    valid_batches = \
    ImageDataGenerator(rescale=1./255).flow_from_directory(valid_path\
                                      ,target_size=(image_height,image_width)\
                                      ,classes=labels,\
                                      batch_size=val_batchsize)
    
    #Genera el batch de prueba
    
    test_batches = \
    ImageDataGenerator(rescale=1./255).flow_from_directory(test_path\
                                     ,target_size=(image_height,image_width)\
                                     ,classes=labels)
        
    # %% Entrenamiento de la red
    
    # Entrena la red
    
    history = model.fit_generator(train_batches\
                        ,steps_per_epoch=\
                            train_batches.samples/train_batches.batch_size\
                        ,validation_data = valid_batches\
                        ,validation_steps = \
                            valid_batches.samples/valid_batches.batch_size\
                        ,epochs = epochs\
                        ,verbose = 1)     
    
    # Guarda el modelo entrenado
    
    model.save(file_model)
    
    # %% Grafica la evolucion de los parametros durante el entrenamiento
    
    # list all data in history
    
    print(history.history.keys())
    
    plt.figure()
    
    # summarize history for accuracy
    
    plt.plot(history.history['acc'])
    
    plt.plot(history.history['val_acc'])
    
    plt.title('model accuracy')
    
    plt.ylabel('accuracy')
    
    plt.xlabel('epoch')
    
    plt.legend(['train', 'valid'], loc='upper left')
    
    plt.show()
    
    plt.savefig('vgg16_v1_acc.png')
        
    # summarize history for loss
    
    plt.figure()
    
    plt.plot(history.history['loss'])
    
    plt.plot(history.history['val_loss'])
    
    plt.title('model loss')
    
    plt.ylabel('loss')
    
    plt.xlabel('epoch')
    
    plt.legend(['train', 'valid'], loc='upper left')
    
    plt.show()
            
    plt.savefig('vgg16_v1_loss.png')
    
    # %% Obtiene los labels reales del set de prueba (TARDA.....)
    
    test_steps = int(np.ceil(test_batches.samples/test_batches.batch_size))
    
    test_labels = np.zeros((test_batches.samples,),dtype = int)
    
    for i in range(test_steps-1):
    
        test_imgs,l = next(test_batches)
    
        l = l[:,1]

        test_labels[int(i*test_batchsize):int((i+1)*test_batchsize)] = l
    
    test_imgs,l = next(test_batches)

    i = i + 1

    l = l[:,1]

    test_labels[int(i*test_batchsize):int(len(test_labels))] = l


    # %% Predice sobre el set de prueba (TARDA......)

    # Utiliza la red entrenada para predecir sobre el conjunto de prueba
    
    predictions = model.predict_generator(test_batches,\
                                          steps = test_steps,\
                                          verbose = 1)
    
    # Como la activacion de la ultima capa es softmax, la salida esta expresada
    #en terminos de probabilidad. En caso que sea mayor a 0.5 se asigna 1 y si
    #es menor se asigna 0
    
    predictions = np.where(predictions > 0.5,1,0)
    
    predictions = predictions[:,1]
    
    # %% Calculo de la matriz de confusion
    
    # Calcula la matriz de confusion usando los labels predichos y los reales
    
    cm = confusion_matrix(test_labels,predictions)
    
    # Categorias que se imprimen en la matriz de confusion
    
    cm_plot_labels = labels
    
    # Imprime la matriz de confusion
    
    plot_confusion_matrix(cm,\
                          cm_plot_labels,\
                          title='confusion matrix'\
                          ,normalize = False)

    # %% Calcula los parametros
    
    TP = cm[1][1] # Verdadero positivo (Predije SI y era SI)
    
    FP = cm[0][1] # Falso positivo (Predije SI y era NO)
    
    TN = cm[0][0] # Verdadero negativo (Predije NO y era NO)
    
    FN = cm[1][0] # Falso negativo (Predije NO y era NO)

    PCP = TP + FP # Cantidad de veces que el algoritmo predijo SI

    PCN = TN + FN # Cantidad de veces que el algoritmo predijo NO

    CP = TP + FN # Cantidad de veces que ha ocurrido SI
    
    CN = FP + TN # Cantidad de veces que ha ocurrido NO

    M = PCP + PCN # Cantidad de ocurrencias

    S = (TP)/(CP) # Que tan posible es que mi algoritmo detecte SI si ocurrio SI
    
    PPV = TP/PCP # Que tan posible es que haya ocurrido SI si predije SI
    
    Acc = (TP+TN)/(M) # Que tan posible es que mi algoritmo le pegue


    # %% Almacena los resultados
    
    file = open(file_results,"w") # Abre el archivo para escritura
    
    file.write('MODELO.\n')
    
    with(redirect_stdout(file)):
        model.summary()
        
    file.write('\n')
               
    file.write('HIPERPARAMETROS.\n\n')
    
    file.write('Learning rate: {:.6f}\n'.format(lr))
    
    file.write('Loss function: {:s}\n'.format(loss))
    
    file.write('Metrics: {:s}\n'.format(metrics))
    
    file.write('Optimizer: {:s}\n'.format(opt))
    
    file.write('Train batch size: {:d}\n'.format(train_batchsize))

    file.write('Valid batch size: {:d}\n'.format(val_batchsize))

    file.write('Test batch size: {:d}\n'.format(test_batchsize))
    
    file.write('Epochs: {:d}\n'.format(epochs))
    
    file.write('Modelo preentrenado: {:s}\n'.format('VGG16'))
    
    file.write('Weights: {:s}\n'.format('ImageNet'))
    
    file.write('Trainable: {}')
    
    file.write('\nENTRENAMIENTO.\n\n')
    
    file.write('Cantidad total: {:d}\n'.format(train_batches.samples))
    
    file.write('Accuracy:\n')
    
    file.write('Loss:\n')
    
    file.write('\nVALIDACION.\n\n')

    file.write('Cantidad total: {:d}\n'.format(train_batches.samples))
    
    file.write('Validation Accuracy:\n')
    
    file.write('Validation Loss:\n')
    
    file.write('\nPRUEBA.\n\n')

    file.write('Cantidad total: {:d}\n'.format(M))

    file.write('Verdadero positivo: {:d}\n'.format(TP))

    file.write('Falso negativo: {:d}\n'.format(FN))
    
    file.write('Verdadero negativo: {:d}\n'.format(TN))

    file.write('Falso positivo: {:d}\n'.format(FP))
                
    file.write('Condición positiva: {:d}\n'.format(CP))
    
    file.write('Condición negativa: {:d}\n'.format(CN))
    
    file.write('Predijo condicion positiva: {:d}\n'.format(PCP))

    file.write('Predijo condicion positiva: {:d}\n'.format(PCN))

    file.write('Sensibilidad: {:.6f}\n'.format(S))

    file.write('Valor predictivo positivo: {:.6f}\n'.format(PPV))

    file.write('Exactitud: {:.6f}\n'.format(Acc))
    
    file.close()
    
# %% Ejecuta el testbench

testbench()