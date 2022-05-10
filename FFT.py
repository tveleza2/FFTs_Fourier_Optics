import numpy as np
import matplotlib.pyplot as plt

def phase( fv , size):
   ''' The omega term in DFT and IDFT formulas'''
   return np.exp2(-(2 * np.pi * 1j * fv) / size)



def fft(x):
   ''' FFT of 1-d signals
   usage : X = fft(x)
   where input x = list containing sequences of a discrete time signals
   and output X = dft of x '''

   n = len(x)
   if n == 1:
      return x
   Feven, Fodd = fft(x[0::2]), fft(x[1::2])
   combined = [0] * n
   for m in range(int(n/2)):
     combined[m] = combined[m] + Feven[m] + phase(m,n) * Fodd[m]
     combined[int(m + n/2)] = combined[int(m + n/2)] + Feven[m] - phase(m,n) * Fodd[m]
   return combined





x = np.linspace(-2*np.pi,2*np.pi,512)
y = np.cos(x)
F = fft(y)
plt.plot(x,y)
plt.show()
plt.plot(x,np.real(F))
plt.show()