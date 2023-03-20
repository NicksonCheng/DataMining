import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
radius = [6.0, 5.0, 4.4, 3.5, 3.6]
min_density = [15, 13, 14, 13, 14]
c = 1


def calculate_dis(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


def find_candidate(candidate_idx, all_distance, target_idx, visit):
    candidate_idx.append(target_idx)
    visit[target_idx] = 1
    target = all_distance[target_idx]
    neighbors_idx = np.where(target <= radius[c-1])[0]
    if (np.size(neighbors_idx) >= min_density[c-1]):
        for idx in neighbors_idx:
            if (visit[idx] == 0):
                find_candidate(candidate_idx, all_distance, idx, visit)
    return candidate_idx, visit


if __name__ == "__main__":

    path = "./Clustering_test"
    while (os.path.exists(f"./{path}{c}")):
        x = []
        y = []
        with open(f"./{path}{c}") as file:
            for line in file:
                data = line.split(' ')
                x.append(float(data[0]))
                y.append(float(data[1]))
        x = np.array(x)
        y = np.array(y)
        total = x.shape[0]
        all_distance = np.zeros(shape=(total, total))
        for i in range(total):
            for j in range(total):
                if (i == j):
                    continue
                distance = calculate_dis(x[i], y[i], x[j], y[j])
                all_distance[i][j] = distance
        # target_idx=random.randint(0,total)
        visit = np.zeros(total)

        all_point_neighbors = []
        for point in all_distance:
            target_idx = np.where(point < radius[c-1])[0]
            neighbors_size = np.size(target_idx)
            all_point_neighbors.append(neighbors_size)

        # every time the start point is from central point
        all_point_neighbors = np.array(all_point_neighbors)
        max_neighbors_idx = np.argmax(all_point_neighbors)
        dataset_id = np.ndarray(total)
        cluster_id = 1
        while True:
            if (np.all(visit == 1)):
                break
            candidate_idx = []
            candidate_idx, visit = find_candidate(
                candidate_idx, all_distance, max_neighbors_idx, visit)

            # find another central point
            visit_idx = np.where(visit == 1)[0]
            all_point_neighbors[visit_idx] = 0
            max_neighbors_idx = np.argmax(all_point_neighbors)

            # recorder the candidate cluster id

            dataset_id[candidate_idx] = cluster_id
            plt.scatter(x[candidate_idx], y[candidate_idx])
            cluster_id += 1

        # save the result
        with open(f"output_{c}.txt", "w") as outfile:
            for i in range(total):
                outfile.write(f"{x[i]} {y[i]} {dataset_id[i]}\n")
        plt.savefig(f"Cluster_{c}.jpg")
        plt.clf()
        c += 1
