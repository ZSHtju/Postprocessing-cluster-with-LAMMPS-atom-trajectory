'''
Map water molecule IDs to their corresponding cluster IDs at each timestep using Freud cluster analysis.

Input: ./dumpdata/step_{timestep}.txt (LAMMPS dump files)
Output: ./ClusterP/.txt (clusterID - atomID mapping with position flags)

Author: Shenghui Zhong, Boxuan Cui (Beihang University)
'''

import os
import MDAnalysis as mda
import freud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def computeCondenseClusterSize(n_start, step):
    
    print('hello!!!') 
    n_frames = 0
    if os.path.exists('dumpdata'):
        for file_name in os.listdir('dumpdata'):
            if file_name.endswith('.txt'):
                n_frames += 1  #count number of '.txt', output step 
    print("n_frames = {}".format(n_frames))
    timeSeries = np.arange(n_start, n_start+n_frames*step, step)

    for nf, time in enumerate(timeSeries):
        u = mda.Universe('./dumpdata/step_{}.txt'.format(time), format="LAMMPSDUMP", lammps_coordinate_convention="unscaled",   atom_style="id mol type x y z vx vy vz", lengthunit="Angstrom", timeunit="ps")
        
        Oframe = u.select_atoms("type 1")
        mWframe = u.select_atoms("type 3")
        n_atoms = len(Oframe)
        print(f'number of O atom = {n_atoms}')
        atom_indices = np.arange(1, n_atoms + 1).reshape(n_atoms, 1)

        cl = freud.cluster.Cluster()       
        cl_props = freud.cluster.ClusterProperties()      
        dl = freud.cluster.Cluster()       
        dl_props = freud.cluster.ClusterProperties()  
        
        cl.compute((Oframe.dimensions[:3], np.concatenate((Oframe.positions, mWframe.positions),axis=0)), neighbors={'r_max':3.36}) # the cutoff distance is 3.36 Å
        dl.compute((Oframe.dimensions[:3], Oframe.positions[:]), neighbors={'r_max':3.36})    
             
        cl_props.compute((Oframe.dimensions[:3], np.concatenate((Oframe.positions, mWframe.positions),axis=0)), cl.cluster_idx)        
        dl_props.compute((Oframe.dimensions[:3], Oframe.positions[:]), dl.cluster_idx)
              
        dlfa = pd.DataFrame(np.column_stack((dl.cluster_idx, atom_indices)),columns=['cluster_idx', 'atom_idx']) # atomIndex of oxygen atoms in each water molecule, note that the atom index starts from 1.
        dfca = dlfa.groupby("cluster_idx") # get the oxygen atom index in the same cluster
        
        clusterIndex = []
        atomIndex = []
        inClProps = []

        for i in range(dl_props.centers.shape[0]):
            is_in_cl_props = 1
            if dl_props.centers[i] in cl_props.centers:
                is_in_cl_props = 0
            df_key = dfca.get_group(i)
            for a in df_key['atom_idx']:
                clusterIndex.append(i)
                atomIndex.append(a)
                inClProps.append(is_in_cl_props)

        if not os.path.exists('./ClusterP'):
            os.makedirs('./ClusterP')
        result = np.column_stack([np.array(clusterIndex), np.array(atomIndex), np.array(inClProps)])
        np.savetxt("./ClusterP/{}.txt".format(nf), result, delimiter=' ', fmt='%d')

step = 10000
computeCondenseClusterSize(20000,step)