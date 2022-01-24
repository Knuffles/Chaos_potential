import numpy as np
import matplotlib.pyplot as plt
import random as r
import math as m
from numpy.linalg import norm 


#w vecteur dans l'espace des phases => w=[x,y,vx,vy]
def majpos(t,w):
    new_w=np.zeros(4)
    R=np.sqrt(w[0]**2+w[1]**2)
    new_w[0]=w[2]
    new_w[1]=w[3]
    new_w[2]=-w[0]/R**3
    new_w[3]=-w[1]/R**3
    return(new_w)

def majpos_HH(t,w):
    new_w=np.zeros(4)
    new_w[0]=w[2]
    new_w[1]=w[3]
    new_w[2]=-(w[0]+2*w[0]*w[1])
    new_w[3]=-(w[1]+w[0]**2-w[1]**2)
    return(new_w)

def Euler(h,n,f,w0,t0):
    L=np.zeros((n,4))
    K=np.zeros(n)
    P=np.zeros(n)
    L[0,:]=w0
    t=t0
    T=[t]
    for i in range(n-1):
        t=t+h
        T.append(t)
        L[i+1]=L[i]+h*f(t,L[i])
        K[i]=kineticEnergy(L[i])
        P[i]=potentialEnergy(L[i])
    E = K+P
    return(L,E)
    
def RK2(h,n,f,w0,t0):
    L=np.zeros((n,4))
    K=np.zeros(n)
    P=np.zeros(n)
    L[0,:]=w0         #liste des vecteurs 
    l=0               #valeurs intermédiaires
    t=t0
    T=[t]
    for i in range(n-1):
        t=t+h
        T.append(t)
        l=L[i]+0.5*h*f(t,L[i])
        L[i+1]=L[i]+h*f(t+0.5*h,l)
        K[i]=kineticEnergy(L[i])
        P[i]=potentialEnergy(L[i])
    E = K+P
    return(L,E)
    
def RK4(h,n,f,w0,t0):
    L=np.zeros((n,4))
    K=np.zeros(n)
    P=np.zeros(n)
    L[0,:]=w0
    t=t0
    T=[t]
    for i in range(n-1):
        t=t+h
        k1=f(t,L[i])
        k2=f(t+h/2,L[i]+k1*h/2)
        k3=f(t+h/2,L[i]+k2*h/2)
        k4=f(t+h,L[i]+h*k3)
        L[i+1]=L[i]+(k1+2*k2+2*k3+k4)*h/6
        K[i]=kineticEnergy(L[i])
        P[i]=potentialEnergy(L[i])
        T.append(t)
    E = K+P
    return(L)
    
def lightRK4(h,f,w0,t0):
    t=t0+h
    k1=f(t,w0)
    k2=f(t+h/2,w0+k1*h/2)
    k3=f(t+h/2,w0+k2*h/2)
    k4=f(t+h,w0+h*k3)
    w=w0+(k1+2*k2+2*k3+k4)*h/6
    return(w,t)

def kineticEnergy(w):
    v = (w[2]**2 + w[3]**2)**0.5
    return(0.5*v**2)

def potentialEnergy(w):
    R = (w[0]**2 + w[1]**2)**0.5
    return (1/R)


def maxX(E):
    return((2*E)**0.5)

def initialConditions(E):
    maxx=maxX(E)
    x=r.uniform(-maxx,maxx)
    y=0
    vymax = (2*E - x**2)**0.5
    vy = r.uniform(0,vymax)
    vx = (2*E - vy**2 - x**2)**0.5
    return np.array((x,y,vx,vy))

def poincarreSection(L):
    sectionVY = []
    sectionY = []
    for k in range(np.shape(L)[0]-1):
        if (L[k,0]*L[k+1,0])<0:
            sectionY.append(L[k,1]-L[k,0]*(abs(L[k+1,1]-L[k,1]))/(abs(L[k+1,0]-L[k,0])))
            sectionVY.append(L[k,3]-L[k,0]*(abs(L[k+1,3]-L[k,3]))/(abs(L[k+1,0]-L[k,0])))
    return sectionY,sectionVY
            


if __name__ == '__main__':
     H=[]
     E=[]
     L=[]
     T=[]
     Erk4=[]
     Erk2=[]
     Eeuler=[]
     Llight=[]
     sectionY=[]
     sectionV=[]
     
     #trace orbits
#     w=[1,0,0,1]
#     h=1e-3
#     Eul,E=Euler(h,50000,majpos,w,0)
#     rk2,E=RK2(h,50000,majpos,w,0)
#     rk4=RK4(h,50000,majpos,w,0)
#     plt.plot(Eul[:,0],Eul[:,1],label='euler')
#     plt.plot(rk2[:,0],rk2[:,1], label='rk2')
#     plt.plot(rk4[:,0],rk4[:,1], label='rk4')
#     plt.legend()
#     plt.title('Integration of a circular orbit')
     #end trace orbits

    #trace energy variation /!\ add E to RK4
#     w=[1,0,0,1]
#     for h in range(1,100):
#       h=h*1e-3
#       print(h)
#       L,E=Euler(h,int(10//h),majpos,w,0)
#       Eeuler.append(abs(E[-2]-E[0]))
#       L,E=RK2(h,int(10//h),majpos,w,0)
#       Erk2.append(abs(E[-2]-E[0]))
#       L,E=RK4(h,int(10//h),majpos,w,0)
#       Erk4.append(abs(E[-2]-E[0]))
#       H.append(h)
#        
#     plt.plot(H,Eeuler,label='euler')
#     plt.plot(H,Erk2, label='rk2')
#     plt.plot(H,Erk4, label='rk4')
#            
#     plt.legend()
#
#     plt.xscale('log')
#     plt.xlabel('timestep')
#     plt.ylabel('energy')
#     plt.yscale('log')
    #end trace energy variation
    
    
     w= initialConditions(1/12)
     L = RK4(1e-3,10000,majpos_HH,w,0)
     wl=w
     tl=0
     for i in range(10000):
         wl,tl=lightRK4(1e-3,majpos_HH,wl,tl)
         Llight.append(norm(L[i,:])-norm(wl))
         if not (norm(L[i,:])-norm(wl)):
             print(norm(L[i,:])-norm(wl))
         #tester la différence entre les deux RK4
         sectionY,sectionV = poincarreSection(L)
    
     plt.scatter(sectionY,sectionV,marker='+')
     plt.xlabel('position along y')
     plt.ylabel('speed along y')
    
    
