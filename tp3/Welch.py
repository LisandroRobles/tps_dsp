import numpy as np

N = 1000
L = np.array([500, 200, 100,  50,  20], dtype=int)
D = np.array(L/2, dtype=int)
K = np.array(2 * (N/L) - 1, dtype=int)
mu = 0
sigma2 = 2

rep = 200
sesgo = np.zeros(len(K))
varianza = np.zeros(len(K))
noise = np.random.normal(mu, np.sqrt(sigma2), (rep,N)) #Realizaciones

for i in range(len(K)):
#    W = np.array(BartlettW(L[i])).reshape(1,L[i])
    W = np.array(BartlettW(L[i]))
    U = 0
    for n in range(len(W)):
        U += abs(W[n])**2
    U /= len(W)
    
    W = W.reshape(1,L[i])
    PSDw = np.zeros([rep,L[i]])
    for k in range(rep):
        n1 = np.array(noise[k,:])
        x = np.zeros([K[i],L[i]])
        for j in range(K[i]):
            beg = j*D[i]
            fin = int( (j/2 + 1)*L[i] )
            x[j,:] = n1[beg:fin]*W
        
        PSDx = (abs(np.fft.fft(x,axis=1))**2)/L[i]
        PSDw[k,:] = np.mean(PSDx,axis=0)
    
    PSDw /= U    
    PSDp = np.mean(PSDw,axis=0)
    PSDvar = np.var(PSDw,axis=0)
    varianza[i] = np.mean(PSDvar,axis=0)
    sesgo[i] = sigma2 - np.mean(PSDp)

PrintArb(K,varianza,'Número de bloques - K','varianza','Periodograma de Welch - Solapamiento 50% - Ventana bartlett')
PrintArb(K,sesgo,'Número de bloques - K','sesgo','Periodograma de Welch - solapamiento 50% - Ventana bartlett')