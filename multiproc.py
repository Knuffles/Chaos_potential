# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:09:57 2021

@author: 33649
"""


import numpy as np
import multiprocessing as mp
import Kepler as kp
import matplotlib.pyplot as plt
import os 


def random_init_RK4(E):
    w=kp.initialConditions(E)
    L = kp.RK4(1e-3,100000,kp.majpos_HH,w,0)
    return(L)



def collect_result(result):
    results.append(result)
    return(result)


if __name__ == '__main__':
    global results
    sectionY = []
    sectionVY=[]
    Y=[]
    V=[]
    pool=mp.Pool(6)
    results = []
    pool.apply_async(random_init_RK4, (1/12,), callback=collect_result)
    
    pool.close()
    pool.join()
    
    print(len(results[0]))
    
    for k in range(len(results)):
        Y,V = kp.poincarreSection(results[k])
        sectionVY += V
        sectionY += Y
        print(len(sectionVY))
    plt.scatter(sectionY,sectionVY,marker='+')
    plt.xlabel('position along y')
    plt.ylabel('speed along y')

