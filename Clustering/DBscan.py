import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
def calculate_dis(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

if __name__ == "__main__":
    x=[]
    y=[]
    random.seed(time.time())
    with open("Clustering_test1") as file:
        for line in file:
            data=line.split(' ')
            x.append(float(data[0]))
            y.append(float(data[1]))
    x=np.array(x)
    y=np.array(y)
    total=x.shape[0]
    radius=30
    min_density=15
    all_distance=np.zeros(shape=(total,total))
    for i in range(total):
        for j in range(total):
            if(i==j): continue
            distance=calculate_dis(x[i],y[i],x[j],y[j])
            all_distance[i][j]=distance
    all_distance=np.sort(all_distance,axis=0)
    #target_idx=random.randint(0,total)
    visit=np.zeros(total)
    count=0
    while count<total:    
        target_idx=480
        target=all_distance[target_idx]
        neighbors_idx=np.where(target<=radius)
        neighbors=target[neighbors_idx]

    # plt.scatter(x,y)
    # plt.show()