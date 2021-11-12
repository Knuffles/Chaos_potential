# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:09:57 2021

@author: 33649
"""

import numpy as np
import multiprocessing as mp
import Kepler as kp
import matplotlib as plt

def random_init_RK4(i, E):
    w=kp.initialConditions(E)
    L,T,e = kp.RK4(1e-3,100000,kp.majpos_HH,w,0)
    return(L)

pool=mp.Pool(mp.cpu_count()//2)
results = []
E=[0.001,0.002,0.003]

def collect_result(result):
    #global results
    #results.append(result)
    return(result)


if __name__ == '__main__':
    for i in range(10):
        r = pool.apply_async(random_init_RK4, args=(i, 0.01), callback=collect_result)
        print("iteration "+str(i))
        results.append(r.get())
        #print(results)

    pool.close()
    pool.join()
    print(results)
    # for k in range(np.size(results)):
    #     sectionY,sectionV = kp.poincarreSection(results[k])
    # plt.scatter(sectionY,sectionV,marker='+')
    # plt.xlabel('position along y')
    # plt.ylabel('speed along y')

#print(random_init_RK4(0.01))
