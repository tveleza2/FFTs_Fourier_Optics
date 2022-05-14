import numpy as np
import FFT
from matplotlib import pyplot as plt
def imageShow (inp, title):
    '''
    # Function to display an image
    # Inputs:
    # inp - The input complex field
    # title - The title of the displayed image        
    '''
    plt.imshow(inp, cmap='gray'), plt.title(title)  # image in gray scale
    plt.show()  # show image

    return

def normalize(A):
    mini = np.amin(A)
    A = A - mini
    maxi = np.amax(A)
    A = A / maxi
    return A

def Angular_Spectrum(Uin, z, x, y,wavelength):
    Uspec = FFT.FFT2(np.fft.fftshift(Uin))
    Uspec = normalize(Uspec)
    fx = np.fft.fftshift(np.fft.fftfreq(x))
    fy = np.fft.fftshift(np.fft.fftfreq(y))
    FX,FY = np.meshgrid(fx,fy)
    phase = np.exp(2j*z*np.pi*np.sqrt(np.power(1/wavelength, 2) - np.power(FX, 2) - np.power(FY, 2)))
    Uout = Uspec * phase
    imageShow(np.abs(Uout),'ABS')
    Uout= FFT.FFT2(np.fft.ifftshift(Uout))
    return Uout

