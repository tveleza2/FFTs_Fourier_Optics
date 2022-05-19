import numpy as np
from Propagator import np_as as AS
from matplotlib import pyplot as plt
import FFT as F
from zernike import RZern
from scipy import interpolate
import pdb

def phase (inp):
    '''
    # Function to calcule the phase representation of a given complex field using the 
    #function 'angle'
    # Inputs:
    # inp - The input complex field
    '''        
    out = np.arctan2(np.imag(inp),np.real(inp))
    # out = np.angle(inp)
    return out

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

    plt.imshow(inp, cmap='viridis'), plt.title(title)  # image in gray scale
    
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

    plt.imshow(inp1, cmap='viridis'), plt.title(title)  # image in gray scale
    
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
    cart = RZern(Zord)
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


n = 256
dx = 1e-5
l = n*dx
d = 1*10**(-3)
nzer = 6
Z = zerniks(nzer,n)

wavelength = 655 * 10**(-9)

hres = 16
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
imageShow(F.normalize(intensity(Uout)), hres=g , title = 'Hartmann Patern Plain Wavefront normalized')


cenplain, CHplain = Hartmann_centroids(Uout,hres)


Uin = np.zeros((n,n))
# Uin = Z[4]
ap = np.linspace(1,0.01,nzer)
for i in range(nzer):
    Uin = Uin + ap[i]*Z[i]
Uin = Uin / np.linalg.norm(ap)

# Uin = np.ones((n,n))
# Uin = Uin * np.exp(2j * np.pi * r / wavelength )/r 


ABERRADO = intensity(Uin)
Uin = H * Uin
imageShow(F.normalize(intensity(Uin)), hres=g , title = 'Aberrated wavefront')

Uout = hn(intensity(AS(Uin,z,wavelength,dx,dx)),hres)


cenaber, CHaber = Hartmann_centroids(Uout,hres)

fig = plt.figure()
imageShow2(CHaber+CHplain,g,fig,title='Hartmann pattern aberrated')
# imageShow2(CHplain,g,fig,title='Hartmann pattern aberrated')
plt.show()

x = y = np.arange(n)
X,Y = np.meshgrid(x,y)

dWtemp = (cenaber - cenplain) / z
# pdb.set_trace()


dWtempx = interpolate.interp2d(np.ravel(cenplain[0]),np.ravel(cenplain[1]),np.ravel(dWtemp[0]),kind='linear')
dWtempx = np.ravel(dWtempx(x,y))

dWtempy = interpolate.interp2d(np.ravel(cenplain[0]),np.ravel(cenplain[1]),np.ravel(dWtemp[1]),kind='linear')
dWtempy = np.ravel(dWtempy(x,y))

dW = np.append(dWtempx,dWtempy)
# pdb.set_trace()



dzx = dzy = np.empty_like(Z,dtype='complex128')
dztemp = np.zeros((n,n),dtype='complex128')
dztot = np.empty((2*n**2,nzer),dtype='complex128')
for i in range(nzer):
    fx = fy = np.fft.fftshift(np.fft.fftfreq(n))
    FX,FY = np.meshgrid(fx,fy)
    dzx[i] = IFFT2((2j*np.pi*FX*FFT2(Z[i])))
    # for j in range(hres):
    #     for q in range(hres):
    #         dztemp[j,q] = dzx[i][int(cenplain[0,j,q]),int(cenplain[1,j,q])]
    dztemp = dzx[i].copy()
    dzte = np.ravel(dztemp)

    # pdb.set_trace()
    dzy[i] = IFFT2((2j*np.pi*FY*FFT2(Z[i])))
    # for j in range(hres):
    #     for q in range(hres):
    #         dztemp[j,q] = dzy[i][int(cenplain[0,j,q]),int(cenplain[1,j,q])]
    dztemp = dzy[i].copy()
    dz = np.append(dzte,np.ravel(dztemp))
    for j in range(2*n**2):
        dztot[j,i] = dz[j].copy()


aux = np.matmul(dztot.T , dztot)
a = np.linalg.inv(aux) @ dztot.T @ dW
a = abs(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(a))))
# pdb.set_trace()   
print(a)
sol = np.zeros((n,n))
for i in range(nzer):
    sol = sol + a[i]*Z[i]
# sol = FFT2(sol)
sol = F.normalize(intensity(sol))
imageShow(ABERRADO,g,'Incident Wavefront')
imageShow(sol,g,'Reconstruction with polynimials')





