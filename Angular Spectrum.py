import numpy as np
import FFT


def Angular_Spectrum(Uin, z, x, y,wavelength):
    Uspec = np.fft.fftshift(FFT.FFT2(np.fft.fftshift(Uin)))
    fx = np.fft.fftfreq(x)
    fy = np.fft.fftfreq(y)
    phase = np.exp(2j*z*np.pi*np.sqrt((1/wavelength)**2 - np.power(fx, 2) - np.power(fy, 2)))
    Uout = Uspec * phase
    Uout= np.fft.ifftshift(FFT.FFT2(np.fft.ifftshift(Uout)))
    return Uout






