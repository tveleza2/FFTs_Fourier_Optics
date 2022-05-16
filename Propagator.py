import numpy as np
import FFT
import pdb


def Angular_Spectrum(Uin, z, x, y, wavelength, dx=1, dy=1):
    '''
    # Function to propagate a complex field by the Angular spectrum methode
    Inputs:

    Uin - The input complex field

    z - The propagation distance
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


def RS_1(Uin, z, dx=1, dy=1, wavelength=633e-9):
    (Nmax,Mmax) = np.shape(Uin)
    x = np.linspace(-dx*Nmax/2,dx*Nmax/2,Nmax)
    y = np.linspace(-dy*Mmax/2,dy*Mmax/2,Mmax)
    X,Y = np.meshgrid(x,y)
    k=2*np.pi/wavelength
    Uspec=np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(Uin)))
    r = np.sqrt(np.ones((Nmax,Mmax))*z**2+np.power(X,2)+np.power(Y,2))
    h = (z/(1j*wavelength))*np.exp(1j*k*r)/np.power(r,2)
    H = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(h)))
    Uout = Uspec * H
    Uout = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(Uout)))
    # pdb.set_trace()
    return Uout





