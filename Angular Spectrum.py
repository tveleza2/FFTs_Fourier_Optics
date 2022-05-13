import numpy as np
import FFT


def Angular_Spectrum(Uin, z, x, y):
    Uspec = np.fft.fftshift(FFT.FFT2(np.fft.fftshift(Uin)))
    fx = np.fft.fftfreq()
    fy = np.fft.fftfreq(y)

    Uout=0
    return Uout






