import numpy as np
from Propagator import RS_1, Angular_Spectrum as AS
from matplotlib import pyplot as plt
import FFT as F

def intensity(inp, log=False):
    '''
    # Function to calcule the intensity representation of a given complex field
    # Inputs:
    # inp - The input complex field
    # log - boolean variable to determine if a log representation is applied
    '''
    out = np.abs(inp)
    out = out*out
    if log == True:
        out = 20 * np.log(out)
    return out

def imageShow (inp, title):
    '''
    # Function to display an image
    # Inputs:
    # inp - The input complex field
    # title - The title of the displayed image        
    '''
    plt.imshow(inp, cmap='jet'), plt.title(title)  # image in gray scale
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

def rect2(X,Y,W,H):
    (xmax,ymax) = X.shape
    IN = 0*X
    for j in range (xmax):
        for i in range (ymax):
            if (np.abs(X[j,i]) < W) and (np.abs(Y[j,i]) < H):
                IN[j,i] = 1
    return IN

# Changing parameters for this simulation
r = 0.0002 # radius of the apperture in meters
n = 256 # Number of divisions per axis on every domain
z = 0.8 # propagation distance in meters
vw = 0.005 # Window of visualization, mantaining an aspect ratio between de distance propagated and the width of the patern
wavelength = 655 * 10**(-9) # Wavelength of the light

size = [-vw,vw]
d = (size[1]-size[0])/n
x = y = np.linspace(size[0],size[1],n)

X,Y = np.meshgrid(x,y)

Aperture = circ2(X,Y,r)
# Aperture = rect2(X,Y,1*r,2*r)
Illumination = np.exp(-1j*2*np.pi*z/wavelength)
Uin = Aperture * Illumination
A = AS(Uin,z,n,n,wavelength,d,d)
Aint = F.normalize(intensity(A))
R = RS_1(Uin,z,d,d,wavelength)
Rint = F.normalize(intensity(R))

# imageShow(Aint,'AS')
# imageShow(Rint,'RS1')
color = 'gray'
# %%
fig = plt.figure(figsize=(9,8))

ax0 = fig.add_subplot(2, 2, 1)
im = ax0.imshow(Aperture,cmap=color,extent=[size[0],size[1],size[0],size[1]])
ax0.set_title('Apperture')
ax0.set_xlabel('x')
ax0.set_ylabel('y')
fig.colorbar(im)

ax1 = fig.add_subplot(2, 2, 2)
im = ax1.imshow(Aint,cmap=color,extent=[size[0],size[1],size[0],size[1]])
ax1.set_title('Angular Spectrum')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
fig.colorbar(im)

ax1 = fig.add_subplot(2, 2, 3)
im = ax1.imshow(Rint,cmap=color,extent=[size[0],size[1],size[0],size[1]])
ax1.set_title('RS1')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
fig.colorbar(im)

ax1 = fig.add_subplot(2, 2, 4)
im = ax1.imshow(np.abs(Aint-Rint),cmap=color,extent=[size[0],size[1],size[0],size[1]])
ax1.set_title('Absolute difference')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
fig.colorbar(im)

fig.tight_layout()

plt.title('First Problem, Second Exam, Fourier Optics')
plt.savefig('D:\OneDrive - Universidad EAFIT\Semestre VII\Fourier Optics\Parcial2\Punto1\Punto1.jpg',dpi=1200)
plt.show()

