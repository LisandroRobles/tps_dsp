#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:31:02 2019

@author: lisandro

En este script se grafica en tiempo y en frecuencia a la wavelet madre a
diferentes escalas 

"""
# %% Inclusion de paquetes

import numpy as np #Paquetes para calculos numericos

import pywt #Paquete para wavelets

import matplotlib.pyplot as plt #Paquete para graficar

# %% Testbench

def testbench():
    
    # %% Preparacion del entorno
        
    plt.close('all') #Cierra las ventanas abiertas
    
    # %% Parametros de la wavelet

    wav = pywt.ContinuousWavelet('gaus2') #Tipo de wavelet madre

    width = wav.upper_bound - wav.lower_bound #Ancho de la wav en s = 1

    scales = [1, 2, 4, 8, 16, 32] #Escalas en que se evalua la wavelet

    max_len = int(np.max(scales)*width + 1) #Longitud de la wav a la max escala
    
    t = np.arange(max_len) #Vetor temporal
    
    # %% Grafica en tiempo y espectro a diferentes resultados
    
    #Rango de la wavelet en escala = 1
    print("Continuous wavelet will be evaluated over the range [{}, {}]".format(wav.lower_bound, wav.upper_bound))

    #Genera la ventana
    fig, axes = plt.subplots(len(scales), 2, figsize=(12, 6))
    
    for n, scale in enumerate(scales): #Para cada escala

        # The following code is adapted from the internals of cwt
        int_psi, x = pywt.integrate_wavelet(wav, precision=10)
        
        step = x[1] - x[0]
        
        j = np.floor(np.arange(scale * width + 1) / (scale * step))
        
        if np.max(j) >= np.size(int_psi):
        
            j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
        
        j = j.astype(np.int)

        # normalize int_psi for easier plotting
        int_psi /= np.abs(int_psi).max()

        # discrete samples of the integrated wavelet
        filt = int_psi[j][::-1]

        # The CWT consists of convolution of filt with the signal at this scale
        # Here we plot this discrete convolution kernel at each scale.

        nt = len(filt)
        
        t = np.linspace(-nt//2, nt//2, nt)
        
        axes[n, 0].plot(t, filt.real, t, filt.imag)
        
        axes[n, 0].set_xlim([-max_len//2, max_len//2])
        
        axes[n, 0].set_ylim([-1, 1])
        
        axes[n, 0].text(50, 0.35, 'scale = {}'.format(scale))

        f = np.linspace(-np.pi, np.pi, max_len)
        
        filt_fft = np.fft.fftshift(np.fft.fft(filt, n=max_len))
        
        filt_fft /= np.abs(filt_fft).max()
        
        axes[n, 1].plot(f, np.abs(filt_fft)**2)
        
        axes[n, 1].set_xlim([-np.pi, np.pi])
        
        axes[n, 1].set_ylim([0, 1])
        
        axes[n, 1].set_xticks([-np.pi, 0, np.pi])
        
        axes[n, 1].set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        
        axes[n, 1].grid(True, axis='x')
        
        axes[n, 1].text(np.pi/2, 0.5, 'scale = {}'.format(scale))

        axes[n, 0].set_xlabel('time (samples)')
        
        axes[n, 1].set_xlabel('frequency (radians)')
        
        axes[0, 0].legend(['real', 'imaginary'], loc='upper left')
        
        axes[0, 1].legend(['Power'], loc='upper left')
        
        axes[0, 0].set_title('filter')
        
        axes[0, 1].set_title(r'|FFT(filter)|$^2$')
        
# %% Ejecuta el testbench
        
testbench()