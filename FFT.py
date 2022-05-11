import numpy as np

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

   f, m, n = pad2(f)
   return np.transpose(FFT(np.transpose(FFT(f)))), m, n

def NUFFT(F,X):
   N = len(X)
   R = 1
   M_r = R*N
   k = np.linspace(-N/2,N/2,N)
   out = 0
   return out
