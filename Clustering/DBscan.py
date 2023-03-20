import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
visit=[]
radius=100.
min_density=15
def calculate_dis(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)
def find_candidate(candidate_idx,all_distance,target_idx):
    target=all_distance[target_idx]
    neighbors_idx=np.where(target<=radius)
    if(np.size(neighbors_idx)>=min_density):
        candidate_idx.append(target_idx) 
        print(neighbors_idx)
        for idx in neighbors_idx:
            find_candidate(candidate_idx,all_distance,idx)
    return candidate_idx
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
    all_distance=np.zeros(shape=(total,total))
    for i in range(total):
        for j in range(total):
            if(i==j): continue
            distance=calculate_dis(x[i],y[i],x[j],y[j])
            all_distance[i][j]=distance
    #target_idx=random.randint(0,total)
    while True:
        if(np.all(visit==1)):
            break
        target_idx=480
        candidate_idx=[]
        find_candidate(candidate_idx,all_distance,target_idx)
        print(candidate_idx)
        break
            

    # plt.scatter(x,y)
    # plt.show()