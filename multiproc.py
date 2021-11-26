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
import time

def random_init_RK4(E):
    w=kp.initialConditions(E)
    L = kp.RK4(1e-3,1000000,kp.majpos_HH,w,0)
    Y,V = kp.poincarreSection(L)
    R = [Y,V]
    return R



def collect_result(R):
    sectionY.append(R[0])
    sectionVY.append(R[1])
    return(sectionY, sectionVY)


if __name__ == '__main__':
    t1 = time.time()
    results = []
    sectionY = []
    sectionVY=[]
    pool=mp.Pool(7)
    [pool.apply_async(random_init_RK4, (1/100,), callback=collect_result) for k in range(100)]
    pool.close()
    pool.join()
    t2 = time.time()
    print("solve time =", t2-t1)
    for k in range(len(sectionY)):  
        plt.scatter(sectionY[k],sectionVY[k],marker='o', s=1)
    plt.xlabel('position along y')
    plt.ylabel('speed along y')

