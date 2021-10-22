import numpy as np
import matplotlib.pyplot as plt

#w vecteur dans l'espace des phases => w=[x,y,vx,vy]
def majpos(t,w):
    new_w=np.zeros(4)
    R=np.sqrt(w[0]**2+w[1]**2)
    new_w[0]=w[2]
    new_w[1]=w[3]
    new_w[2]=-w[0]/R**3
    new_w[3]=-w[1]/R**3
    return(new_w)

def Euler(h,n,f,w0,t0):
    L=np.zeros((n,4))
    L[0,:]=w0
    t=t0
    T=[t]
    for i in range(n-1):
        t=t+h
        T.append(t)
        L[i+1]=L[i]+h*f(t,L[i])
    return(L,T)
    
def RK2(h,n,f,w0,t0):
    L=np.zeros((n,4))
    L[0,:]=w0         #liste des vecteurs 
    l=0               #valeurs intermédiaires
    t=t0
    T=[t]
    for i in range(n-1):
        t=t+h
        T.append(t)
        l=L[i]+0.5*h*f(t,L[i])
        L[i+1]=L[i]+h*f(t+0.5*h,l)
    return(L,T)
    
def RK4(h,n,f,w0,t0):
    L=np.zeros((n,4))
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
        T.append(t)
    return(L,T)
    

L,T=Euler(0.0001,100000,majpos,[1,0,0,1],0)
plt.plot(L[:,0],L[:,1])

plt.show()






