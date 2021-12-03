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
    
if __name__ == '__main__':
    #initial parameters
    E = 1/6
    eps = 0.00001
    
    
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
    
    T=np.array(t).reshape((-1,1))
    F=np.array(Frac)
    print(np.shape(T))
    print(np.shape(F))
    regr = linear_model.LinearRegression()
    regr.fit(T,F)
    droite = regr.coef_
    
    #tracé des deux trajectoires
#    plt.plot(L[:,0],L[:,1], label='rk4')
#    plt.plot(Leps[:,0],Leps[:,1], label='rk4+eps')
#    plt.legend()
    
    #tracé de la différence entre les deux trajectoires
#    plt.plot(t[40:],DL[40:])
#    plt.loglog()
    
    #tracé de la différence entre les deux trajectoires dans l'espace des phases

    plt.xlabel("log(t)")
    plt.ylabel("log(|D-Deps|)")
    plt.title("E=%f" %E)
    plt.loglog(t[40:],Frac[40:])  
    plt.loglog(t[40:],droite[40:])

    #tracé dans l'espace des phases
#    print(len(sectionV),len(sectionY))
    
#    plt.scatter(sectionY,sectionV,marker='+')
#    plt.scatter(sectionYeps,sectionVeps,marker='+')
#    plt.xlabel('position along y')
#    plt.ylabel('speed along y')
#    

