import numpy as np
import FFT


def Angular_Spectrum(Uin, z, x, y, wavelength, dx=1, dy=1):
    '''
    # Function to propagate a complex Field by the Angular spectrum theory
    # Inputs:
    # Uin - The input complex field
    # z - The propagation distance
    #    
    '''

    Uspec = FFT.FFT2(np.fft.fftshift(Uin))

    fx = np.fft.fftshift(np.fft.fftfreq(x,dx))
    fy = np.flip(np.fft.fftshift(np.fft.fftfreq(y,dy)))
    FX,FY = np.meshgrid(fx,fy) 

    phase = np.exp(2j*z*np.pi*np.sqrt(np.power(1/wavelength, 2) - np.power(FX, 2) - np.power(FY, 2)))
    
    Uout = Uspec * phase
    Uout= FFT.FFT2(np.fft.ifftshift(Uout))
    
    return Uout

