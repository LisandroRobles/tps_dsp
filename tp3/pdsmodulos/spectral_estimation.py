#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:08:56 2018

@author: lisandro
"""

#Paquetes

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as win
import scipy.fftpack as fftpack
from numpy import linalg as LA

#Funciones

def plot_psd(Sxx):
    
    #Genero el vector de frecuencias
    N = np.size(x,0)
    df = (2*np.pi)/N
    f = np.linspace(0,df*(N-1),N)
    
    Sxx = np.transpose(np.array([[np.sqrt(1)*Sij if (Sij != Si[self.fmin] and Sij != Si[self.fmax-1]) else Sij for Sij in Si] for Si in np.transpose(X)],dtype = float))

    return 0
    

def periodogram(x,n1 = 0,n2 = -1):
    
    if n2 == -1:
        n2 = np.size(x,0)
    
    X = fftpack.fft(x[n1:n2],n2-n1,0)
    Xmod = np.abs(X)
    Sxx = np.power(Xmod,2)/(n2-n1) 

    return Sxx
    
    
def modified_periodogram():
    
    return 0

def bartlett():
    
    return 0

def welch():

    return 0
    
def testbench():
    
    #Parametros del muestreo
    fs = 1024
    N = 1024
    Ts = 1/fs
    
    #Vector temporal
    t = np.linspace(0,Ts*(N-1),N)
    n = np.linspace(0,N-1,N)
    
    #Parametros de la señal
    ao = 1
    po = 0
    fo = (2*np.pi)/N
    
    #Generación de la señal
    x = ao*np.sin((fo*n) + po)
    
    #Ploteo de la señal
    plt.figure()
    plt.plot(t,x)
    plt.grid()

    #Periodograma de la señal
    Sxx = periodogram(x,0,np.size(x,0))
    
    #Ploteo del periodograma
    plt.figure()
    plt.plot(Sxx)
    
#Script
    
testbench()