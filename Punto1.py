import numpy as np
from Propagator import Angular_Spectrum as AS
from matplotlib import pyplot as plt
import FFT as F


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

def circ2(X, Y, radius):
    '''
    Makes a 2D circ function.
    Size is the length of the signal
    pradius is the pradius of the circ function
    '''
    (xmax,ymax) = X.shape
    IN = 0*X
    for j in range (xmax):
        for i in range (ymax):
            if np.power(X[j,i], 2) + np.power(Y[j,i], 2) < np.power(radius, 2):
                IN[i,j] = 1
    return IN

# Changing parameters for this simulation
r = 0.01 # radius of the apperture in meters
n = 512 # Number of divisions per axis on every domain
z = 600 # propagation distance in meters
vw = z/3000 # Window of visualization, mantaining an aspect ratio between de distance propagated and the width of the patern
wavelength = 655 * 10**(-9) # Wavelength of the light

size = [-vw,vw]
d = (size[1]-size[0])/n
x = y = np.linspace(size[0],size[1],n)

X,Y = np.meshgrid(x,y)

Aperture = circ2(X,Y,r)
imageShow(Aperture,'Aperture')
Illumination = np.exp(-1j*2*np.pi*z/wavelength)
Uin = Aperture * Illumination
output = F.normalize(np.abs(AS(Uin,z,n,n,wavelength,d,d)))
imageShow(output,'Propagated Light')
