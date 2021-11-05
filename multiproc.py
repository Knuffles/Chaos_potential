import numpy as np
import multiprocessing as mp
import Kepler as kp

def random_init_RK4(E):
    x,y,vx,vy=kp.initialConditions(E)
    w=[x,y,vx,vy]
    print(w)
    L,T,e = kp.RK4(1e-3,100000,kp.majpos_HH,w,0)
    print(L)
    return(L)


pool=mp.Pool(mp.cpu_count())
results = []

def collect_result(result):
    global results
    results.append(result)
    
for i in range(50):
    pool.apply_async(random_init_RK4, args=(0.01), callback=collect_result)

pool.close()
pool.join()
print(results)
print("bonjour")

#print(random_init_RK4(0.01))
