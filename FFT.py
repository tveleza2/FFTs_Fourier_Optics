from math import pi
import numpy as np
import matplotlib.pyplot as plt

def FFT(x):
    """
    A recursive implementation of 
    the 1D Cooley-Tukey FFT, the 
    input should have a length of 
    power of 2. 
    """
    N = len(x)
    
    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X

def pad2(x):
   m, n = np.shape(x)
   M, N = 2 ** int(np.ceil(np.log(m, 2))), 2 ** int(np.ceil(np.log(n, 2)))
   F = np.zeros((M,N), dtype = x.dtype)
   F[0:m, 0:n] = x
   return F, m, n

def FFT2(f):
   '''FFT of 2-d signals/images with padding
   usage X, m, n = fft2(x), where m and n are dimensions of original signal'''
#    f, m, n = pad2(f)
   return np.fft.ifftshift(np.transpose(DFT(np.transpose(DFT(f)))))
   
def DFT(x,cords=None,frecs=None,modes='U'):
    """
    Function to calculate the 
    discrete Fourier Transform 
    of a 1D real-valued signal x
    """
    
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))/N
    var = k*n
    if modes != 'U':
        N = len(frecs)
        k = np.reshape(frecs,(N,1))
        print('k =',k)
        n = cords
        print('n =',n)
        var = np.multiply(k,n)

    e = np.exp(-2j * np.pi * var)
    
    X = np.dot(e, x)
    
    return X

def NUFFT(Func,X):
   '''Function to calculate de DFT of a group of not evenly sampled
   1D groups of data'''
   Xmax = np.amax(X)
   df = 1/Xmax
   mdx = mindis(X)
   Fmax = 1/mdx
   N = int(Fmax/df)
   Frecs = np.linspace(0,Fmax,N)
   out = DFT(Func,X,Frecs,'NU')
   return out,Frecs

def NUFFT2(Func,X,Y):
    return np.transpose(NUFFT(np.transpose(NUFFT(Func,X)),Y))

def mindis(V):
    d = abs(np.amax(V))
    for i in range(len(V)-1):
        dtemp = abs(V[i]-V[i+1])
        if dtemp < d:
            d = dtemp
        
    return d

def rect(x):
    y = np.zeros_like(x)
    for i in range(len(y)):
        if abs(x[i])<0.5:
            y[i]=1
    return y

x = np.linspace(0,2*np.pi,1000)
y = rect(x)
FT, f = NUFFT(y,x)
plt.plot( np.abs(np.fft.fftshift(FT)))
plt.show()
