import Kepler as kp
import matplotlib.pyplot as plt
import numpy as np
import epsilon 
from numpy.linalg import norm 
import multiprocessing as mp

#returns 0 if non chaotic and 1 if chaotic
def chaosDiscrim(E,eps):
    #random initial conditions
    w = kp.initialConditions(E)
    #initial condition with small variation
    weps = epsilon.epsilonInit(w,eps)
    h=1e-3
    n=1000000
    Frac=[]
    t=[]
    
    #orbits calculation
#    L = kp.RK4(1e-3,1000000,kp.majpos_HH,w,0)
#    Leps = kp.RK4(1e-3,1000000,kp.majpos_HH,weps,0)
    time=0
    #orbits calculation using light RK4
    for i in range(n):
        new_w,new_time=kp.lightRK4(h,kp.majpos,w,time)
        new_w_eps,new_time_eps=kp.lightRK4(h,kp.majpos,weps,time)
        w=new_w
        weps=new_w_eps
        time=new_time
        Frac.append(abs(norm(w)-norm(weps)))
        t.append(time)

    #slope of the difference fonction
    a,b=epsilon.slope(t,Frac,9000,120000)
    if (a<1):
        return(0)
    else:
        return(1)

def collect_result(R):
    traj.append(R)
    return(traj)

if __name__ == '__main__':
    E=np.linspace(0.01,1,11) #pas besoin aller plus haut que 1/6 => reste inutile
    nb=10 #devrait etre entre 30 et 50
    eps = 0.00001
    print(E)
    chaostest=np.zeros(len(E))
    
    print(chaostest)
    
    for e in range(len(E)):
        #nombre d'orbites chaotiques
        print(E[e])
        traj=[]
        pool=mp.Pool(7)
        #applies chaosDiscrim on nb random trajectories
        [pool.apply_async(chaosDiscrim, (E[e],eps,), callback=collect_result) for k in range(nb)]
        pool.close()
        pool.join()
        chaos=sum(traj)
        print(chaos)
        #fraction chaotique sur 10 tests par energie
        chaostest[e]=chaos/nb
    
    
    plt.xlabel("E")
    plt.ylabel("Fraction Chaotique")
    plt.title("Fraction chaotique des energies")
    plt.plot(E,chaostest)
