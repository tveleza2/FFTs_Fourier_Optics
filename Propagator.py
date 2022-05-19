from attr import field
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

def np_as(field, z, wavelength, dx, dy):
    '''
    # Function to diffract a complex field using the angular spectrum approach
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx/dy - sampling pitches
    '''
    M, N = field.shape
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)
    
    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)
        
    phase = np.exp2(1j * z * np.pi * np.sqrt(np.power(1/wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))
	
    tmp = field_spec*phase
    
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)
	
    return out


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





