# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:53:20 2021

@author: 33649
"""


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
    eps = 0
    
    
    #random initial conditions
    w = kp.initialConditions(E)
    #initial condition with small variation
    eps = 0.1 * w[0]
    weps = epsilonInit(w,eps)
    print(w[0])
    print(weps[0])
    L=[]
    Leps=[]
    DL=[]
    t=[]
    sectionV=[]
    sectionY=[]
    sectionVeps=[]
    sectionYeps=[]
    
    L = kp.RK4(1e-3,2000000,kp.majpos_HH,w,0)
    Leps = kp.RK4(1e-3,2000000,kp.majpos_HH,weps,0)
    
    for i in range(len(L)):
        DL.append(np.sqrt((L[i,0]-Leps[i,0])**2 + (L[i,1]-Leps[i,1])**2))
        t.append(i)
    
    sectionY,sectionV = kp.poincarreSection(L)
    sectionYeps, sectionVeps = kp.poincarreSection(Leps)
    
    #tracé des deux trajectoires
#    plt.plot(L[:,0],L[:,1], label='rk4')
#    plt.plot(Leps[:,0],Leps[:,1], label='rk4+eps')
#    plt.legend()
    
    # tracé de la différence entre les deux trajectoires
    plt.plot(t,DL)
    
    #tracé dans l'espace des phases
    print(len(sectionV),len(sectionY))
    
    plt.scatter(sectionY,sectionV,marker='+')
    plt.scatter(sectionYeps,sectionVeps,marker='+')
    plt.xlabel('position along y')
    plt.ylabel('speed along y')