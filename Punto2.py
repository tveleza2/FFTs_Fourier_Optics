import numpy as np
from Propagator import RS_1, np_as as AS
from matplotlib import pyplot as plt
import FFT as F
from zernike import Rzern
import pdb



def circ2D(size, pradius, center=None):
    '''
    Makes a 2D circ function.
    Size is the length of the signal
    pradius is the pradius of the circ function
    '''
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    data = np.zeros((size,size),dtype='float')
    for j in range (size):
        for i in range (size):
            if np.power(j-x0, 2) + np.power(i-y0, 2) < np.power(pradius, 2):
                data[i,j] = 1
    return data

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

def Hartman_Mask(X,Y,n,r):
    xtemp = int(X/n)
    ytemp = int(Y/n)
    # X = X -(X % xtemp)
    # Y = Y -(Y % ytemp)
    Uout = np.zeros((X,Y))
    
    for i in range(n):
        for j in range(n):
            Uout[xtemp*i:xtemp*(i+1),ytemp*j:ytemp*(j+1)] = circ2D(xtemp,r)
    return Uout,X,xtemp

def imageShow (inp, hres, title='Image'):
    '''
    # Function to display an image
    # Inputs:
    # inp - The input complex field
    # title - The title of the displayed image        
    '''
    fig = plt.figure()
    axes = fig.add_subplot(111)

    plt.imshow(inp, cmap='gray'), plt.title(title)  # image in gray scale
    
    major_ticks = np.arange(0, inp.shape[0], hres)
    minor_ticks = np.arange(0, inp.shape[1], hres)

    axes.set_xticks(major_ticks)
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_yticks(major_ticks)
    axes.set_yticks(minor_ticks, minor=True)
    #axes.set_yticks(major_ticks)
    #axes.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    plt.grid(which='both')

    plt.show()  # show image

    return

def imageShow2 (inp1, hres, fig, title='Image'):
    '''
    # Function to display an image
    # Inputs:
    # inp - The input complex field
    # title - The title of the displayed image        
    '''
    axes = fig.add_subplot(111)

    plt.imshow(inp1, cmap='gray'), plt.title(title)  # image in gray scale
    
    major_ticks = np.arange(0, inp1.shape[0], hres)
    minor_ticks = np.arange(0, inp1.shape[1], hres)

    axes.set_xticks(major_ticks)
    axes.set_xticks(minor_ticks, minor=True)
    axes.set_yticks(major_ticks)
    axes.set_yticks(minor_ticks, minor=True)
    #axes.set_yticks(major_ticks)
    #axes.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    plt.grid(which='both')
    return

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

def hn(inp,href):
    Uout = inp * 0
    (xmax,ymax)=np.shape(inp)
    xmax = int(xmax/href)
    ymax = int(ymax/href)
    for i in range(href):
        for j in range(href):
            Uout[xmax*i:xmax*(i+1),ymax*j:ymax*(j+1)] = F.normalize(inp[xmax*i:xmax*(i+1),ymax*j:ymax*(j+1)])
    return Uout

def centroids(inp):
    cx = cy = 0
    (xmax,ymax) = inp.shape
    c1 = np.arange(xmax)
    c2 = np.arange(ymax)
    C1,C2 = np.meshgrid(c1,c2)
    cx = (inp * C1).sum()/inp.sum()
    cy = (inp * C2).sum()/inp.sum()
    return cx,cy

def Hartmann_centroids(inp,href):
    imagen = inp * 0
    Uoutx = np.empty((href,href))
    Uouty = np.empty((href,href))
    (xmax,ymax)=np.shape(inp)
    xmax = int(xmax/href)
    ymax = int(ymax/href)
    for i in range(href):
        for j in range(href):
            Uoutx[i,j] , Uouty[i,j] = centroids(inp[xmax*i:xmax*(i+1),ymax*j:ymax*(j+1)])
            imagen[xmax*i:xmax*(i+1),ymax*j:ymax*(j+1)] = circ2D(xmax,hr,(Uoutx[i,j],Uouty[i,j]))
            Uoutx[i,j] += xmax*i
            Uouty[i,j] += ymax*j
    Uout = np.array([Uoutx,Uouty])
    return Uout,imagen

def zerniks(Zord,size):
    Zord = 6
    cart = Rzern(Zord)
    ddy = np.linspace(-1.0, 1.0, size)
    xv, yv = np.meshgrid(ddy, ddy)
    cart.make_cart_grid(xv, yv)
    c = np.zeros(cart.nk)
    Zpols=[]
    for i in range(0,Zord):
        c*=0
        c[i]=1
        cc=cart.eval_grid(c,matrix=True)
        cc=np.nan_to_num(cc,0)
        Zpols.append(cc)
    return Zpols


def FFT2(inp):
    inp = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(inp)))
    return inp

def IFFT2(inp):
    inp = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(inp)))
    return(inp)

n = 1024
dx = 1e-5
l = n*dx
d = 1*10**(-3)

wavelength = 655 * 10**(-9)

hres = 8
hr = 6
z = 0.008

H,n,g = Hartman_Mask(n, n, hres, hr)

x = y = np.linspace(-l/2 , l/2,n)
X,Y = np.meshgrid(x,y)
r = np.sqrt(d**2 + X**2 + Y**2)


Uin = np.ones((n,n))
Uin = Uin * np.exp(2j * np.pi * z / wavelength )
Uin = H * Uin


imageShow(F.normalize(intensity(Uin)), hres=g , title = 'Plain wavefront')

Uout = hn(intensity(AS(Uin,z,wavelength,dx,dx)),hres)


cenplain, CHplain = Hartmann_centroids(Uout,hres)


Uin = np.ones((n,n))
Uin = Uin * np.exp(2j * np.pi * r / wavelength ) / r
Uin = H * Uin
imageShow(F.normalize(intensity(Uin)), hres=g , title = 'Aberrated wavefront')

Uout = hn(intensity(AS(Uin,z,wavelength,dx,dx)),hres)


cenaber, CHaber = Hartmann_centroids(Uout,hres)

fig = plt.figure()
imageShow2(CHaber+CHplain,g,fig,title='Hartmann pattern aberrated')
# imageShow2(CHplain,g,fig,title='Hartmann pattern aberrated')
plt.show()




dW = (cenaber - cenplain) / z
print(dW)
# pdb.set_trace()









