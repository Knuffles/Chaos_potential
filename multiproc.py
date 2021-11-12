import numpy as np
import multiprocessing as mp
import Kepler as kp

def random_init_RK4(i, E):
    x,y,vx,vy=kp.initialConditions(E)
    w=[x,y,vx,vy]
    L,T,e = kp.RK4(1e-3,100000,kp.majpos_HH,w,0)
    return(L)

pool=mp.Pool(mp.cpu_count())
results = []
E=[0.001,0.002,0.003]

def collect_result(result):
    #global results
    #results.append(result)
    return(result)
    
for i in range(10):
    r = pool.apply_async(random_init_RK4, args=(i, 0.01), callback=collect_result)
    print("iteration "+str(i))
    results.append(r.get())
    #print(results)

pool.close()
pool.join()
print(results)


#print(random_init_RK4(0.01))
