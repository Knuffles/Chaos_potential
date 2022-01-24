import Kepler as kp
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm 
from sklearn import linear_model

#add a small variation e to the initial condition in x
def epsilonInit(w,e):
    w2=np.copy(w)
    w2[0]=w2[0]+e
    return(w2)

#return the slope and the intercept of a given trajectory
def slope(t,f,tmin,tmax):
    T=np.array(np.log(t[tmin:tmax])/np.log(10)).reshape((-1,1))
    F=np.array(np.log(f[tmin:tmax])/np.log(10))
    regr = linear_model.LinearRegression()
    regr.fit(T,F)
    a= regr.coef_
    b=regr.intercept_
    return(a[0],b)
    
    
if __name__ == '__main__':
    #initial parameters
    E = 1/100
    eps = 0.00001
    tmin=9000
    tmax=120000
    
    
    #random initial conditions
    w = kp.initialConditions(E)
    #initial condition with small variation
    weps = epsilonInit(w,eps)
    L=[]
    Leps=[]
    DL=[]
    Frac=[]
    t=[]
    sectionV=[]
    sectionY=[]
    sectionVeps=[]
    sectionYeps=[]
    
    L = kp.RK4(1e-3,1000000,kp.majpos_HH,w,0)
    Leps = kp.RK4(1e-3,1000000,kp.majpos_HH,weps,0)
    
    sectionY,sectionV = kp.poincarreSection(L)
    sectionYeps, sectionVeps = kp.poincarreSection(Leps)
    
    for i in range(len(L)):
        DL.append(np.sqrt((L[i,0]-Leps[i,0])**2 + (L[i,1]-Leps[i,1])**2))
        Frac.append(abs(norm(L[i,:])-norm(Leps[i,:])))
        t.append(i)
    
    tlog=np.log(t[tmin:tmax])/np.log(10)
    Fraclog=np.log(Frac[tmin:tmax])/np.log(10)
    
    a,b=slope(t,Frac,tmin,tmax)
    RegDroit=[]
    for i in tlog:
        RegDroit.append(b+a*i)

    
    #both trajectories

#    plt.plot(L[:,0],L[:,1], label='rk4')
#    plt.plot(Leps[:,0],Leps[:,1], label='rk4+eps')
#    plt.legend()
#    plt.ylabel("y")
#    plt.xlabel("x")
    #end both trajectories
#    
    #tracé de la différence entre les deux trajectoires
#    plt.plot(t[40:],DL[40:])
#    plt.loglog()
    
    #trajectory difference in phase space
    plt.xlabel("log(t)")
    plt.ylabel("log(|D-Deps|)")
    plt.title("a=%f" %a)
    plt.plot(tlog,Fraclog)  
    plt.plot(tlog,RegDroit)  
    #end trajectory difference in phase space

    #trajectory in phase space
#    print(len(sectionV),len(sectionY))
    
#    plt.scatter(sectionY,sectionV,marker='+')
#    plt.scatter(sectionYeps,sectionVeps,marker='+')
#    plt.xlabel('position along y')
#    plt.ylabel('speed along y')
#    #end rajectory in phase space

