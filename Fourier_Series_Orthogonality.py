
import numpy as np
import math as mt
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

nump=16
v_0=5
w_0=2*np.pi*v_0
T_0=1/v_0
alpha=5
Result_Matrix=np.zeros([nump,nump])

t=np.linspace((alpha*T_0),(alpha+1)*T_0,10000)
deltat=t[-1]/len(t)


for n in range(1,nump+1):
    for m in range(1,nump+1):
        sumador=0
        for ti in t:
            sumador = np.exp((1j*w_0*ti)*(n-m))*deltat+sumador
        Result_Matrix[n-1][m-1]=round(np.abs(sumador*v_0))
imageShow(Result_Matrix,'Ortogonalidad')