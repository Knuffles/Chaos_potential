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
    return(E)
    
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
    return(E)
    
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
    return(E)

def kineticEnergy(w):
    v = (w[2]**2 + w[3]**2)**0.5
    return(0.5*v**2)

def potentialEnergy(w):
    R = (w[0]**2 + w[1]**2)**0.5
    return (1/R)
H=[]  
E=[]
Erk4=[]
Erk2=[]
Eeuler=[]
w=[1,0,0,1]
for h in range(1,10000):
    h=h*1e-5
    print(h)
    E=Euler(h,int(10//h),majpos,w,0)
    Eeuler.append(abs(E[-2]-E[0]))
    E=RK2(h,int(10//h),majpos,w,0)
    Erk2.append(abs(E[-2]-E[0]))
    E=RK4(h,int(10//h),majpos,w,0)
    Erk4.append(abs(E[-2]-E[0]))
    H.append(h)
    
plt.plot(H,Eeuler,label='euler')
plt.plot(H,Erk2, label='rk2')
plt.plot(H,Erk4, label='rk4')

plt.legend()
plt.xscale('log')
plt.yscale('log')







