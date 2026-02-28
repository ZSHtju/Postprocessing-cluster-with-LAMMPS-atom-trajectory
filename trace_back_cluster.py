'''
Script for tracking long-lived condensed clusters on base:
Identifies clusters with lifetime >13 waters,
Tracks their ID/size/location (0/1/2) over time,
Calculates their displacement and trajectory.

Input files:
./ClusterP/*.txt: cluster composition (clusterID - atomID pairs)
cluster_properties.txt: cluster properties with location marks (0/1/2)
./dumpdata/step_{timestep}.txt: LAMMPS trajectory files

Output files:
cluster_evolution.txt: evolution of all tracked clusters
./Cluster_evolution_target/cluster*.txt: trajectory data for individual clusters

software: MDAnalysis, Freud, NumPy, Pandas
Author: Shenghui Zhong, Boxuan Cui (Beihang University)
'''

import os
import pandas as pd
import numpy as np
from collections import Counter

#step 1 # Identify critical clusters (> 13 waters) and track their evolution over time
folder_path = './ClusterP' # Get all cluster composition files (clusterID - atomID pairs for each frame)
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt') and f.split('.')[0].isdigit()]
if not txt_files:
    print("no txt")
else:
    file_numbers = [int(f.split('.')[0]) for f in txt_files]
    max_number = max(file_numbers)
final_file = np.loadtxt('./ClusterP/{}.txt'.format(max_number))
stepSeries = np.arange(0, max_number+1)[::-1]

first_column = final_file[:, 0]
counts = Counter(first_column)
cluster_critical = [num for num, count in counts.items() if count > 13]
cluster_critical = np.array(cluster_critical)

results = stepSeries.reshape(-1, 1)

print(cluster_critical)
for id in cluster_critical:
    cluster_atom = final_file[final_file[:, 0] == id, 1]
    temp_results_list = []
    
    for step in stepSeries:
        data = np.loadtxt('./ClusterP/{}.txt'.format(step))
        
        cluster_from = 0
        max_common = 0
        cluster_num = 0
        cluster_ids = np.unique(data[:, 0])

        for cluster_id in cluster_ids:
            cluster_n_data = data[data[:, 0] == cluster_id]
            atom_id = cluster_n_data[:, 1]
            common_count = len(np.intersect1d(atom_id, cluster_atom))
            if common_count > max_common:
                max_common = common_count
                cluster_from = cluster_id
                cluster_num = len(cluster_n_data)
        position = np.mean(data[data[:, 0] == cluster_from, 2])
        cluster_atom = data[data[:, 0] == cluster_from, 1]
        temp_results_list.append([cluster_from, cluster_num, position])
    
    temp_results = np.array(temp_results_list)
    results = np.hstack((results, temp_results))

    print(f"{id} finished")

final_results = results[::-1]
header = 'step ' + ' '.join([f'from_{int(id)} num_{int(id)} p_{int(id)}' for id in cluster_critical])
np.savetxt('cluster_evolution.txt', final_results, fmt='%d', header=header, comments='')


#step 2 Calculate displacement and trajectory of selected critical clusters

n_files = 0
for file_name in os.listdir('dumpdata'):
    if file_name.endswith('.txt'):
        n_files += 1  #count number of '.txt', output step 

all_data = np.loadtxt('./cluster_evolution.txt', skiprows=1)
Cluster_positions = np.loadtxt('./cluster_properties.txt') # Contains: time, clusterID, size, center_xyz, position(0/1/2)

timeSeries = np.arange(0, n_files-1, 1)

l_x = 520
l_y = 520

cluster_number = int((all_data.shape[1] - 1) / 3)

for idx in range(cluster_number):
    idx_begin = 3*idx + 1
    idx_end =  idx_begin + 3
    Cluster_ids = all_data[:, idx_begin:idx_end]
    cluster_move = []

    cluster_size = Cluster_ids[:, 1]    # Cluster size evolution
    cluster_id = Cluster_ids[:, 0]      # Cluster ID evolution
    cluster_local = Cluster_ids[:, 2]   # 0 is on the sky, 1 is on the surface
    
    # Find first frame where cluster size >= 3
    start_idx = 0
    for i in range(len(cluster_size)):
        if np.all(cluster_size[i:] >= 3):
            size_start_idx = i
            break
    
    # Find first frame where cluster is on surface (position=1)
    target_cluster_position = []
    for i in range(len(cluster_size)):
        if cluster_local[i] == 1:
            local_start_idx = i
            break

    start_idx = max(size_start_idx, local_start_idx)
    step = (start_idx+2)*10000
    mask_target = (Cluster_positions[:, 0] == step) & (Cluster_positions[:, 1] == Cluster_ids[start_idx, 0])
    target_cluster_position = Cluster_positions[mask_target][0, 3:6]
    print(target_cluster_position)

    # Track cluster movement from beginning to end
    for i in range(1, len(Cluster_ids)):
        if i < start_idx + 1:
            distance = 0
            cluster_center2 = [0, 0, 0]
            z_coor2 = 0
        
        else:
            step2 = (i+2)*10000
            mask2 = (Cluster_positions[:, 0] == step2) & (Cluster_positions[:, 1] == Cluster_ids[i,0])
            cluster_center2 = Cluster_positions[mask2][:, 3:6]

            target_cluster_position = target_cluster_position.flatten()
            cluster_center2 = cluster_center2.flatten()

            # Load atom IDs in this cluster
            all_cluster_atom_id2 = np.loadtxt('./ClusterP/{}.txt'.format(i))
            cluster_atom_id2 = all_cluster_atom_id2[all_cluster_atom_id2[:, 0] == Cluster_ids[i,0], 1]
            # Load dump file to get actual atomic positions
            atom_positions2 = np.loadtxt('./dumpdata/step_{}.txt'.format(step2), skiprows=9)
            cluster_atom_positions2 = atom_positions2[np.isin(atom_positions2[:, 0], cluster_atom_id2), 5]
            z_coor2 = np.mean(cluster_atom_positions2, axis=0)

            x_ij = cluster_center2[0] - target_cluster_position[0]
            y_ij = cluster_center2[1] - target_cluster_position[1]

            if x_ij < -l_x/2:
                x_ij = x_ij + l_x
            elif x_ij > l_x/2:
                x_ij = x_ij - l_x
            if y_ij < -l_x/2:
                y_ij = y_ij + l_y
            elif y_ij > l_y/2:
                y_ij = y_ij - l_y
            
            distance = np.sqrt(x_ij**2 + y_ij**2) # + z_ij**2

        cluster_move.append([i, cluster_id[i], cluster_size[i], cluster_center2[0], cluster_center2[1], z_coor2, distance, cluster_local[i]])
    print(f'cluster{idx} finished')
    
    cluster_move = np.array(cluster_move)
    if not os.path.exists('./Cluster_evolution_target'):
            os.makedirs('./Cluster_evolution_target')
    
    np.savetxt('./Cluster_evolution_target/cluster{}.txt'.format(idx+1), cluster_move, fmt='%d %d %d %f %f %f %f %d', header='step id size x y z dis position', comments='')



