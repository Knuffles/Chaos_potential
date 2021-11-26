import Kepler as kp
import matplotlib.pyplot as plt
import numpy as np

#add a small variation e to the initial condition in x
def epsilonInit(w,e):
    w2=np.copy(w)
    w2[0]=w2[0]+e
    return(w2)
    
if __name__ == '__main__':
    #initial parameters
    E = 1/12
    eps = 1
    
    
    #random initial conditions
    w = kp.initialConditions(E)
    #initial condition with small variation
    weps = epsilonInit(w,eps)
    L=[]
    Leps=[]
    DL=[]
    t=[]
    sectionV=[]
    sectionY=[]
    sectionVeps=[]
    sectionYeps=[]
    
    L = kp.RK4(1e-3,50000,kp.majpos_HH,w,0)
    Leps = kp.RK4(1e-3,50000,kp.majpos_HH,weps,0)
    
    for i in range(len(L)):
        DL.append(np.sqrt((L[i,0]-Leps[i,0])**2 + (L[i,1]-Leps[i,1])**2))
        t.append(i)
    
    sextionY,sectionV = kp.poincarreSection(L)
    sextionYeps, sectionVeps = kp.poincarreSection(Leps)
    
    #tracé des deux trajectoires
#    plt.plot(L[:,0],L[:,1], label='rk4')
#    plt.plot(Leps[:,0],Leps[:,1], label='rk4+eps')
#    plt.legend()
    
    #tracé de la différence entre les deux trajectoires
#    plt.plot(t,DL)
    
    #tracé dans l'espace des phases
    print(len(sectionV),len(sectionY))
    
    plt.scatter(sectionY,sectionV,marker='+')
    plt.scatter(sectionYeps,sectionVeps,marker='+')
    plt.xlabel('position along y')
    plt.ylabel('speed along y')
    

