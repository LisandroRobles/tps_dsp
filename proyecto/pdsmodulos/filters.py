#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:26:10 2018

@author: lisandro
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
import scipy.signal as sig


def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    

    return z, p, k

def moving_average(N):
    
    b = (1/N)*np.ones((N,),dtype = float)
    a = np.zeros((N,),dtype = float)
    a[0] = 1
    
    return b,a

def fir_from_impulse_response(h):
    
    N = np.size(h)
    b = h
    a = np.zeros((N,),dtype = float)
    a[0] = 1
    
    return b,a

def input_delay(N,i):
    
    b = np.zeros((N+1,),dtype = float)
    b[0] = 1
    b[N] = i
    a = np.zeros((N+1,),dtype = float)
    a[0] = 1
    
    return b,a

def matched_filter(x,template,Vumbral):
        
    #Paso el template a punto flotante
    template = np.array(template,dtype = float)
    
    #Los filtros del coeficiente FIR se obtienen invirtiendo en el tiempo
    #el template
    fir_coeffs = template[::-1]

    #Aplico el filtro FIR
    det = sig.lfilter(fir_coeffs,1.0,x,axis = 0)
        
    #Plancho a cero todo lo negativo
    aux = np.zeros((np.size(det),1),dtype = float)
    aux[det > 0] = det[det > 0]
    det = aux
    
    #Normalizo la deteccion
    det = det/np.max(det)
    
    #Lo elevo al cuadrado
    det = np.power(det,2)
    
    #Aplico el umbral
    aux = np.zeros((np.size(det),1),dtype = float)
    aux[det > Vumbral] = det[det > Vumbral]
    det = aux
    
    return det

def decimate(x,fs1,fs2):
    
    #Antes de submuestrearhay que aplicar un filtro anti-allias
    #con frecuencia de corte en la nueva frecuencia de Nyquist
    fny1 = fs1/2
    fny2 = fs2/2
    
    fp = fny2/fny1
    fs = (fny2 + 1)/fny1 
    Ripple = 0.5 #dB
    Att = 50 #dB
    
    lpf_sos_butter = sig.iirdesign(wp = fp, ws = fs, gpass = Ripple, gstop = Att, analog=False, ftype='butter', output='sos')
    
    xfilt = sig.sosfiltfilt(lpf_sos_butter,x,axis = 0)
    
    #Una vez muestrado procedo a diezmar a la senial
    muestras_salteadas = int(fs1/fs2)
    N = int(np.size(x,axis = 0))
    
    #Senial decimada
    xd = xfilt[0:N:muestras_salteadas,:]
    
    #Vector de tiempo para la senial decimada
    Nd = np.size(xd,0)
    Ts2 = 1/fs2
    td = np.linspace(0,Ts2*(Nd-1),Nd)
    
    return (td,xd)