#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 10:52:45 2018

@author: lisandro
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as win
import scipy.fftpack as fftpack
from numpy import linalg as LA

class spectrum_analyzer:
    
    def __init__(self,fs = 1024,N = 1024,algorithm = "fft"):
               
        #Frecuencia de muestreo
        self.fs = fs
        #Cantidad de muestras
        self.N = N
        #Resolucion espectral
        self.df = fs/N
        #Frecuencia maxima sin que haya aliasing (normalizada fn = f/df)
        #Depende de la paridad de N
        if (N % 2) == 0:
            self.fmax = int((N/2)+1)
        else:
            self.fmax = int((N+1)/2)
        
        #Frecuencia minima sin que haya aliasing (normalizada fn = f/df )
        self.fmin = int(0)
        #Vector de frecuencias
        self.f = fftpack.fftfreq(self.N,1/(self.fs))
        #Vector de phi
        self.phi = (self.f)*((2*np.pi)/self.fs)
        #Vector de phi normalizado
        self.phi_norm = self.phi/(2*np.pi)
        #Vector de bines 
        self.bin = self.phi_norm*self.N
        #Metodo de transformacion
        self.set_algorithm(algorithm)
        

    def module(self,x,plot = False,db = False,xaxis = 'frequency'):
        
        X = self.transform(x)/(np.size(x,0))
        
        X_mod = np.abs(X)
        
        #Se genera una vector auxiliar de f para solo plotear banda digital
        #Es decir de 0 a fs/2
        
        if xaxis is 'phi_norm':
            f = np.abs(self.phi_norm[self.fmin:self.fmax])
        elif xaxis is 'phi':
            f = np.abs(self.phi[self.fmin:self.fmax])
        elif xaxis is 'bin':
            f = np.abs(self.bin[self.fmin:self.fmax])
        elif xaxis is 'frequency':
            f = np.abs(self.f[self.fmin:self.fmax])
        else:
            f = np.abs(self.f[self.fmin:self.fmax])

        f = np.reshape(f,(np.size(f),1))
        
        #Se genera un vector auxiliar de modulo para plotear solo banda digital
    
        X_mod_aux = X_mod[self.fmin:self.fmax,:]
        #aux = np.array([2*Xi if (Xi != X_mod_aux[self.fmin] and Xi != X_mod_aux[self.fmax]) else Xi for Xi in X_mod_aux],dtype = float)
        aux = np.transpose(np.array([[2*Xij if (Xij != Xi[self.fmin] and Xij != Xi[self.fmax-1]) else Xij for Xij in Xi] for Xi in np.transpose(X_mod_aux)],dtype = float))
        X_mod_aux = aux

        if db is True:
            X_mod_aux = 20*np.log10(X_mod_aux + np.finfo(float).eps) #Unidad: dBW
                
        #Presentacion frecuencia de los resultados de modulo        
        if plot is True:
            plt.figure()
            plt.title('Espectro de modulo')
            
            plt.stem(f,X_mod_aux)
            plt.axis('tight')
            plt.xlabel('f[Hz]')
            plt.ylabel('|X(f)|[V]')
            plt.grid()
            plt.show()
        
        return(f,X_mod_aux)        

    def psd(self,x,plot = False,db = False,xaxis = 'frequency'):

        #Se genera una vector auxiliar de f para solo plotear banda digital
        #Es decir de 0 a fs/2
        
        if xaxis is 'phi_norm':
            f = np.abs(self.phi_norm[self.fmin:self.fmax])
        elif xaxis is 'phi':
            f = np.abs(self.phi[self.fmin:self.fmax])
        elif xaxis is 'bin':
            f = np.abs(self.bin[self.fmin:self.fmax])
        elif xaxis is 'frequency':
            f = np.abs(self.f[self.fmin:self.fmax])
        else:
            f = np.abs(self.f[self.fmin:self.fmax])

        f = np.reshape(f,(np.size(f),1))
        
        X = self.transform(x)
        X = X/np.sqrt((np.size(X,0)))
        X = np.abs(X)

        #Se genera un vector auxiliar de modulo para plotear solo banda digital
    
        X = X[self.fmin:self.fmax,:]
        #aux = np.array([2*Xi if (Xi != X_mod_aux[self.fmin] and Xi != X_mod_aux[self.fmax]) else Xi for Xi in X_mod_aux],dtype = float)
        Xaux = np.transpose(np.array([[np.sqrt(1)*Sij if (Sij != Si[self.fmin] and Sij != Si[self.fmax-1]) else Sij for Sij in Si] for Si in np.transpose(X)],dtype = float))
        X = Xaux
        Sx = np.power(np.abs(X),2)
        
        if db is True:
            Sx = 10*np.log10(Sx + np.finfo(float).eps) #Unidad: dBW
                
        #Presentacion frecuencia de los resultados de modulo        
        if plot is True:
            plt.figure()
            plt.title('Espectro de modulo')
            
            plt.stem(f,Sx)
            plt.axis('tight')
            plt.xlabel('f[Hz]')
            plt.ylabel('|X(f)|[V]')
            plt.grid()
            plt.show()
        
        return(f,Sx)     

    def module_phase(self,x):
        
        X = self.transform(x)/(self.N)
        
        X_mod = np.abs(X)
        X_ph = np.angle(X,deg = 'True')
        #Thresholdeo la fase planchando a cero las componentes cuyo modulo sea
        #menor a 0.01. Esto lo hago ya que las componentes 0.0000001 + j0.00001
        #daran valores perceptibles de fase
        X_ph = X_ph*(X_mod >= 0.001)
        X_ph = X_ph*(np.abs(X_ph) >= 0.1)
        
        #Se genera una vector auxiliar de f para solo plotear banda digital
        #Es decir de 0 a fs/2
        
        f_aux = np.abs(self.f[self.fmin:(self.fmax + 1)])
        f_aux = np.reshape(f_aux,(np.size(f_aux),1))
        
        #Se genera un vector auxiliar de modulo para plotear solo banda digital
        
        X_mod_aux = X_mod[self.fmin:(self.fmax + 1),:]
        aux = np.array([2*Xi if (Xi != X_mod_aux[self.fmin] and Xi != X_mod_aux[self.fmax]) else Xi for Xi in X_mod_aux],dtype = float)
        X_mod_aux = aux
        
        X_ph_aux = X_ph[self.fmin:(self.fmax + 1),:]
        
        #fo_estimada = np.sum(X_mod_aux * f_aux,axis = 0)/np.sum(X_mod_aux)
        
        #Presentacion frecuencia de los resultados de modulo        
        plt.figure()
        plt.subplot(2,1,1)
        plt.title('Espectro de modulo')
        
        plt.stem(f_aux,X_mod_aux)
        plt.axis('tight')
        plt.xlabel('f[Hz]')
        plt.ylabel('|X(f)|[V]')
        plt.grid()
        
        #Presentacion frecuencial de los resultados de fase
        plt.subplot(2,1,2)
        plt.title('Espectro de fase')
        plt.stem(f_aux,X_ph_aux)
        plt.legend
        plt.axis('tight')
        plt.xlabel('f[Hz]')
        plt.ylabel('Arg{X(f)}[o]')
        plt.grid()
        
        plt.show()
        
        return(self.f,X_mod,X_ph)
        
    def real_imag(self,x):
        
        X = self.transform(x)
        
        X_re = np.real(X)/(self.N)
        X_re = X_re*(np.abs(X_re) >= 0.001)
        X_im = np.imag(X)/(self.N)
        X_im = X_im*(np.abs(X_im) >= 0.001)
        
        f_aux = np.abs(self.f[self.fmin:(self.fmax + 1)])
        f_aux = np.reshape(f_aux,(np.size(f_aux),1))
        X_re_aux = X_re[self.fmin:(self.fmax + 1),:]
        X_im_aux = X_im[self.fmin:(self.fmax + 1),:]
        
        #Presentacion frecuencial de los resultados reales
        plt.figure()
        plt.subplot(2,1,1)
        plt.title('Parte real')
        plt.plot(f_aux,X_re_aux)
        plt.axis('tight')
        plt.xlabel('f[Hz]')
        plt.ylabel('Re{X(f)}[V]')
        plt.grid()
        
        #Presentacion frecuencial de los resultados imaginarios
        plt.subplot(2,1,2)
        plt.title('Parte imaginaria')
        plt.plot(f_aux,X_im_aux)
        plt.legend
        plt.axis('tight')
        plt.xlabel('f[Hz]')
        plt.ylabel('Im{X(f)}[V]')
        plt.grid()
        
        return(self.f,X_re,X_im)
        
    def PSD(self,x,plot = False,db = False):
        
        X = self.transform(x)
        
        X_mod = np.abs(X)/self.N
        
        PSD = np.power(X_mod,2)
        
        f_aux = np.abs(self.f[self.fmin:(self.fmax + 1)])
        f_aux = np.reshape(f_aux,(np.size(f_aux),1))
        f = f_aux
        
        PSD_aux = PSD[self.fmin:(self.fmax + 1),:]
        aux = np.array([2*PSDi if (PSDi != PSD_aux[self.fmin] and PSDi != PSD_aux[self.fmax]) else PSDi for PSDi in PSD_aux],dtype = float)
        PSD = aux
        
        if db is True:
            PSD_aux = 10*np.log10(PSD + np.finfo(float).eps) #Unidad: dBW
        else:
            PSD_aux = PSD
        
        #Presentacion frecuencial de los resultados de PSD
        if plot is True:
            plt.figure()
            plt.title('Densidad espectral de potencia')
            plt.plot(f_aux,PSD_aux)
            plt.axis('tight')
            plt.xlabel('f[Hz]')
            plt.ylabel('|X(f)|[J/Hz]')
            plt.grid()
        
        return (f,PSD_aux)
        
    def dft(self,x):
                
        #Defino el vector X en el que se almacenaran la salida de la transformada
        #Tendra el mismo largo que x, osea N muestras
        #Las mismas estaran muestreadas en la inversa de la duracion de la senal
        #Osea df = 1/(N*Ts) = fs/N
    
        cantidad_secuencias = np.size(x,axis = 1)
        largo_secuencias = np.size(x,axis = 0)
        
        if largo_secuencias < self.N:
            largo_dft = largo_secuencias
        else:
            largo_dft = self.N
        
        X = np.zeros((largo_dft,cantidad_secuencias),dtype = 'complex')
                
        for l in range(cantidad_secuencias):
        
            for k in range(largo_dft):
                
                X[k,l] = 0
                
                for n in range(largo_dft):
                    
                    arg = (2*np.pi*k*n)/(largo_dft)
                    
                    X[k,l] += np.complex(x[n,l],0)*np.complex(np.cos(arg),-np.sin(arg))
        
        return (X)
    
    def fft(self,x):
                
        #Se calcula la dft de la secuencia mediante el algoritmo de fft
                
        X = fftpack.fft(x,self.N,0)
        
        return (X)
    
    def set_algorithm(self,algorithm = "fft"):
        
        if algorithm == "dft":
            
            self.algorithm = "dft"
            
        elif algorithm == "fft":
            
            self.algorithm = "fft"
            
        else:
            
            print("Algoritmo no implementado.\n")
            print("Se establecera la fft como algoritmo por defecto.\n")
            
            self.algorithm = "fft"
    
    def get_algorithm(self):
                
        return self.algorithm
    
    def transform(self,x):
        
        algorithm = self.get_algorithm()
        
        x = x[0:self.N]
        
        if algorithm == "fft":
        
            X = self.fft(x)
            
        elif algorithm == "dft":
            
            X = self.dft(x)
                
        return X
    
    def set_points(self,N = 1024):
        
        #Modifica la cantidad de puntos
        self.N = N
        #Recalcula la resolucion espectral
        self.df = (self.fs)/(self.N)
        #Frecuencia maxima sin que haya aliasing (normalizada fn = f/df)
        self.fmax = int((self.N/2))
        #Frecuencia minima sin que haya aliasing (normalizada fn = f/df )
        self.fmin = int(0)
        #Vector de frecuencias
        self.f = fftpack.fftfreq(self.N,1/(self.fs))

def periodograma(x,fs,ensemble = False):
    
    #Largo de x
    n = int(np.size(x,0))

    #Enciendo el analizador de espectro
    analizador = spectrum_analyzer(fs,n,"fft")

    #Realizo de forma matricial el modulo del espectro de todas las realizaciones
    (f,Sx) = analizador.psd(x,xaxis = 'phi')

    #Hago el promedio en las realizaciones
    Sxm = np.mean(Sx,1)
    
    #Hago la varianza en las realizaciones
    Sxv = np.var(Sx,1)
    
    if ensemble is True:
        
        Sxm = Sx
        Sxv = 0
    
    return (f,Sxm,Sxv)            

def periodograma_modificado(x,fs,window = 'rectangular',ensemble = False):
    
    #Largo de x
    n = int(np.size(x,0))

    #Genero la ventana
    if window is 'rectangular':
        w = np.ones((n,),dtype = float)
    elif window is 'bartlett':
        w = win.bartlett(n)
    elif window is 'hann':
        w = win.hann(n)
    elif window is 'hamming':
        w = win.hamming(n)
    elif window is 'blackman':
        w = win.blackman(n)
    elif window is 'flat-top':
        w = win.flattop(n)
    else:
        w = np.ones((n,),dtype = float)
    
    w = w.reshape((n,1))/LA.norm(w,axis = 0)

    #Ventaneo los datos y normalizo
    xw = x*w

    #Realizo de forma matricial el modulo del espectro de todas las realizaciones
    (f,Sx,Sxv) = periodograma(xw,fs,ensemble = True)
    Sx = n*Sx
    
    #Hago el promedio en las realizaciones
    Sxm = np.mean(Sx,1)
    
    #Hago la varianza en las realizaciones
    Sxv = np.var(Sx,1)
    
    if ensemble is True:
        
        Sxm = Sx
        Sxv = 0
    
    return (f,Sxm,Sxv)  

def bartlett(x,fs,nsect = 1,ensemble = False):
        
    nsect = int(nsect)
    
    n = np.size(x,0)
    L = int(np.floor(n/nsect))
    
    if L is not 0:
    
        realizaciones = np.size(x,1)
        
        if (L % 2) != 0:
            largo_espectro_un_lado = int((L+1)/2)
        else:
            largo_espectro_un_lado = int((L/2)+1)
        
        Sx = np.zeros([largo_espectro_un_lado,int(realizaciones)],dtype = float)
        
        
        n1 = 0        
        for i in range(0,nsect):
            
            #Obtengo el bloque i-esimo de todas las realizaciones
            xi = x[int(n1):int(n1 + L),:]
            
            #Obtengo el periodograma de todas las realizaciones para este
            #bloque
            (f,Sxi,Sxv) = periodograma(xi,fs,ensemble = True) 
            
            #Los promedio
            Sx = Sx + Sxi/nsect
            
            #Muevo el cursor
            n1 = n1 + L
                    
        #Hago el promedio en las realizaciones
        Sxm = np.mean(Sx,1)
        
        #Hago la varianza en las realizaciones
        Sxv = np.var(Sx,1)

        if ensemble is True:
            
            Sxm = Sx
            Sxv = 0
    
    else:
        
        print('La cantidad de bloques es mayor al largo de las realizaciones')
        f = 0
        Sxm = 0
        Sxv = 0
    
    return (f,Sxm,Sxv)
        
def welch(x,fs,L,window = 'bartlett',overlap = 50,ensemble = False):
        
    #Cantidad de muestras de cada realizacion
    n = np.size(x,0)
    
    if L is not 0:
        
        #Obtiene la cantidad de realizaciones
        realizaciones = np.size(x,1)
        
        #En funcion del largo de cada bloque obtiene el largo que tendria 
        #el espectro de un solo lado
        if (L % 2) != 0:
            largo_espectro_un_lado = int((L+1)/2)
        else:
            largo_espectro_un_lado = int((L/2)+1)
        
        Sx = np.zeros([largo_espectro_un_lado,int(realizaciones)],dtype = float)
     
        n1 = 0
        n0 = int((1-(overlap/100))*int(L))
        nsect = int(1 + np.floor((n-L)/(n0)))
        
        for i in range(0,nsect):
            
            #Obtengo el bloque i-esimo de todas las realizaciones
            xi = x[int(n1):int(n1 + L),:]
            
            #Obtengo el periodograma de todas las realizaciones para este
            #bloque
            (f,Sxi,Sxv) = periodograma_modificado(xi,fs,window = window,ensemble = True) 
            
            #Los promedio
            Sx = Sx + Sxi/nsect
            
            #Muevo el cursor
            n1 = n1 + n0
                
        #Hago el promedio en las realizaciones
        Sxm = np.mean(Sx,1)
        
        #Hago la varianza en las realizaciones
        Sxv = np.var(Sx,1)
        
        if ensemble is True:
        
            Sxm = Sx
            Sxv = 0
    
    else:
        
        print('La cantidad de bloques es mayor al largo de las realizaciones')
        f = 0
        Sxm = 0
        Sxv = 0
    
    return f,Sxm,Sxv
    