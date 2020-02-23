#!/usr/bin/env python
# coding: utf-8

# Boundary conditions:
# $V(x=0,y)=0 , v(x=L,y)=0, V(x,y=0)=0$ and $V(x,y=L)=V_0(x)$.
# We are trying to solve $\nabla^2 V = 0$
# By using the same argument as in example 3.4 we have $Y(y) = Ae^{ky} + Be^{-ky}$ and $X(x) = Csin(kx) + Dcos(kx)$
# $V(x, y)=\left(A e^{k y}+B e^{-k y}\right)(C \sin k x+D \cos k x)$ The difference here is that since $V(x=0,y) = 0 = V(x=L,y)$ the x part of the equation has to be sinousidal. By imposing boundary conditions and the intercnage of variables one arrives at $V(\eta, \xi)=\sum_{n=1}^{\infty} C_{n} \sinh (n \pi \eta) \sin (n \pi \xi)$
# and $V_0(\xi)=\sum_{n=1}^{\infty} C_{n} \sinh (n \pi) \sin (n \pi \xi)$. Now we can use what we have found to expand in fourier coefficients.
# Now those fourier coefficients are given by  $C_{n}=\frac{2}{ \sinh (n \pi)} \int_{0}^{1} V_{0}(\xi) \sin (n \pi \xi) d \xi$
# 
# 
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


#Global constants
dx = 0.001

def integrand(X,V_0, n):
    '''Turns array into integrand
    Input: Array of X values, Potential V, n(which coefficient)
    Output: V_0(xi)*sin(n*pi*xi) array xsize
    '''
    return (2/np.sinh(n*np.pi))*V_0(X)*np.sin(n*np.pi*X)

def num_integration(integrand):
    '''Uses trapezoid to compute an integral
    Input: Array integrand, which is to be integrated
    Output: Integrated array
    '''
    return np.trapz(integrand, dx = dx)

def fourierterm(n, xx, yy,Cn):
    '''Computes each term of the fourier sum on a meshgrid
    Input: xx(meshgrid), yy(meshgrid), n(which term)
    Output: each term of the fourier sum on meshgrid form
    '''
    return Cn*np.sinh(n*np.pi*yy)*np.sin(n*np.pi*xx)

def fourierpotential(N,V_0):
    '''Computes the fourier series up to term N.
    Input: Array for X, N-number of terms included, V-desired potential
    output: 2D array with V(x,y)
    '''
    x = np.arange(0,1,dx)
    C_n = np.zeros(N) # N-1 terms
    X = np.linspace(0,1,int(1/dx))
    Y = X.copy()
    XX,YY = np.meshgrid(X,Y)
    V = fourierterm(0,XX,YY,C_n[0])
    for n in range(1,N):
        C_n[n-1] = num_integration(integrand(x,V_0,n))
        V += fourierterm(n,XX,YY,C_n[n-1])
    return XX,YY,V


# In[ ]:


def checknterms(tol,V_0):
    '''Calculates where the euclidian norm of the calculated V_0 from the series is within a wanted tolerance of V
    input:
        tol: Wanted tolerance(float)
        V_0: expression function for V_0
    output:
        n: The amount of terms needed in the fourier series
    '''
    x = np.arange(0,1,dx)
    y = 1
    Vana = V_0(x)
    n= 1
    C_n = np.zeros(1000)
    C_n[0] = num_integration(integrand(x,V_0,n))
    V0_calc =  fourierterm(n, x, y,C_n[0])
    print(V0_calc.size, Vana.size)
    while (np.linalg.norm(Vana-V0_calc,np.inf))> tol:
        n += 1
        C_n[n-1] = num_integration(integrand(x,V_0,n))
        V0_calc += fourierterm(n, x, y,C_n[n-1])
    return n


# In[ ]:


#tester
def testV(X):
    '''Potentialtester
    input: X, coordinate along y =1
    output: X evaluated with the potential 
    '''
    return np.sin(np.pi*5*X)


def plotter(x,y,V):
    '''Makes a nice 3D plot of the potential
    input:
        x,y: 2D meshgrids
        V: The calculated numerical potential
    '''
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    u = ax.plot_surface(x,y,test, alpha =1, cmap = 'jet')
    ax.contour(x,y,testV(x), zdir = 'y', offset = 1.0, cmap = 'jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('V')
    ax.set_title('Potential V')
    plt.show()


# In[ ]:


def neg_gradient(V,Xx,Yy):
    '''Takes a potential and returns the negative gradient, with a larger separation of points
    input: 
        V: Potential V(Potential evaluated on Xx and Yy)
        Xx: 2D meshgrid(our coordinate system)
        Yy: 2D meshgrid(our coordinate system)
    Output: 
        Ex,Ey: The twodimensional gradient
        XX,Yy: Courser coordinate system
    '''
    Xx = Xx[::20,::20]
    Yy = Yy[::20,::20]
    V = V[::20,::20]
    Ex,Ey = np.gradient(V)
    #Have to take the transposed because of how np.gradient works
    return Ex,Ey,Xx,Yy 


# In[ ]:


n = checknterms(0.001,testV)
print(n)
x,y,test = fourierpotential(160,testV)
plotter(x,y,test)
Ex,Ey,x,y = neg_gradient(test, x,y)

color = np.transpose(Ex)**2+np.transpose(Ey)**2
plt.quiver(x,y, -np.transpose(Ex),-np.transpose(Ey),color,cmap = 'jet')
plt.xlabel("y")


# In[ ]:





# In[ ]:




