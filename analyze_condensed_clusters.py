'''
Script for analyzing condensed clusters on base.
For each time frame, identify clusters and mark their location:
- 1: cluster inside pore
- 0: cluster on surface but outside pore  
- 2: cluster in air (not on surface)

Input: ./dumpdata/step_{timestep}.txt (LAMMPS dump files)
Output: ./cluster_properties.txt (cluster properties with time, ID, size, center coordinates, and position flag)

software: MDAnalysis, Freud
Author: Shenghui Zhong, Boxuan Cui (Beihang University)
'''

import os
import MDAnalysis as mda
import freud
import numpy as np
import pandas as pd
from pathlib import Path
import math as mth

def computeCondenseClusterSize(n_start, step_interval):
    print('begin!')
    
    n_frames = 0
    if os.path.exists('dumpdata'):
        for file_name in os.listdir('dumpdata'):
            if file_name.endswith('.txt'):
                n_frames += 1  #count number of '.txt', output step 
    print("n_frames = {}".format(n_frames))
    timeSeries = np.arange(n_start, n_start+n_frames*step_interval, step_interval)
    # n_frames=sum(1 for e in u.trajectory[::step])
    n2 = np.zeros(n_frames)
    sum_size = np.zeros(n_frames)
    ave_size = np.zeros(n_frames)
    df = pd.DataFrame(data=None,columns=['time', 'clusterID', 'cluster_size', 'cluster_center_x', 'cluster_center_y', 'cluster_center_z', 'position'])

    n_cluster_arrays = {}
    for c_size in range(2,15):
        key = f"n_{c_size}"
        n_cluster_arrays[key] = np.zeros(n_frames)  

    for nf, time in enumerate(timeSeries):
        u = mda.Universe('./dumpdata/step_{}.txt'.format(time), format="LAMMPSDUMP", lammps_coordinate_convention="unscaled",
        atom_style="id mol type x y z vx vy vz",
        lengthunit="Angstrom", timeunit="ps")
        # atom_style follows the order in the dump file. Usually, we need the positions.
        
        frame = u.trajectory[0] #useless
        Oframe = u.select_atoms("type 1")
        mWframe = u.select_atoms("type 3")

        Oframe.positions += np.array([-260, -260, 0])[None, :]
        mWframe.positions += np.array([-260, -260, 0])[None, :]
        
        cl = freud.cluster.Cluster()
        cl_props = freud.cluster.ClusterProperties()

        dl = freud.cluster.Cluster()
        dl_props = freud.cluster.ClusterProperties()
                   
        cl.compute((Oframe.dimensions[:3], np.concatenate((Oframe.positions[:], mWframe.positions[:]),axis=0)),neighbors={'r_max':3.36}) 
        # pick all atoms, so that the water cluster on the surface will be identified as one big cluster, the cutoff distance is 3.36 Å
        dl.compute((Oframe.dimensions[:3], Oframe.positions[:]), neighbors={'r_max':3.36}) # the cutoff distance is 3.36 Å

        cl_props.compute((Oframe.dimensions[:3], np.concatenate((Oframe.positions[:],mWframe.positions[:]),axis=0)), cl.cluster_idx)
        dl_props.compute((Oframe.dimensions[:3], Oframe.positions[:]), dl.cluster_idx)


        el_props_centers = []
        el_props_sizes = []

        fl_props_centers = []
        fl_props_sizes = []

        el_props_clusterID = []
        fl_props_clusterID = []
        
        for i in range(dl_props.centers.shape[0]):
            if dl_props.centers[i] not in cl_props.centers:  # Here the clusters on the carbon surface are calculated
                el_props_centers.append(dl_props.centers[i])
                el_props_sizes.append(dl_props.sizes[i])
                el_props_clusterID.append(i)
            else:
                fl_props_centers.append(dl_props.centers[i])
                fl_props_sizes.append(dl_props.sizes[i])
                fl_props_clusterID.append(i)

        print(cl.compute)
        print(dl_props.centers)
        
        XYstep = 40   #Distance between the centers of two adjacent pores.
        x_start = -240
        x_stop = 240
        y_start = -240
        y_stop = 240
        
        porecenters = [(x, y) for x in range(x_start, x_stop+1, XYstep) for y in range(y_start, y_stop+1, XYstep)]
        #Generate the coordinates of the centre of the pole
        #elcluster_properties = []
        for id, clusterP in enumerate(el_props_centers):
            flag = False
            if el_props_sizes[id] >=2:
                for pore in porecenters: 
                    culster2center = mth.sqrt((clusterP[0] - pore[0])**2+(clusterP[1] - pore[1])**2)
                    if culster2center < 10:
                        df = pd.concat([df, pd.DataFrame([{'time':time,
                        'clusterID':el_props_clusterID[id],
                        'cluster_size':el_props_sizes[id],
                        'cluster_center_x':clusterP[0],
                        'cluster_center_y':clusterP[1],
                        'cluster_center_z':clusterP[2],
                        'position':1}])], ignore_index=True)
                        flag = True
                        break
                if flag == False:
                    df = pd.concat([df, pd.DataFrame([{'time':time,
                    'clusterID':el_props_clusterID[id],
                    'cluster_size':el_props_sizes[id],
                    'cluster_center_x':clusterP[0],
                    'cluster_center_y':clusterP[1],
                    'cluster_center_z':clusterP[2],
                    'position':0}])], ignore_index=True)
            print(f'clusterid={id}')
        #elcluster_properties = np.array(elcluster_properties)

        for id, clusterP in enumerate(fl_props_centers):
            if fl_props_sizes[id] >= 2:
                df = pd.concat([df, pd.DataFrame([{'time':time,
                'clusterID':fl_props_clusterID[id],
                'cluster_size':fl_props_sizes[id],
                'cluster_center_x':clusterP[0],
                'cluster_center_y':clusterP[1],
                'cluster_center_z':clusterP[2],
                'position':2}])], ignore_index=True)

    np.savetxt('./cluster_properties.txt', df, delimiter=' ', fmt = '%s')
    return


computeCondenseClusterSize(20000, 10000)

