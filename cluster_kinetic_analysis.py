'''
perform cluster kinetic analysis to determine the critical nucleus size on a external surface
software: MDAnalysis, Freud
file: LAMMPS atom trajectory
Author: Shenghui Zhong, Zheyu Shi (Beihang University)
'''

import os
import MDAnalysis as mda
import freud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# step 1: get the atom index in the cluster ith 
def computeCondenseClusterSize(filename,step):
    
    print('hello!!!') 
    
    u = mda.Universe(filename, format="LAMMPSDUMP", atom_style="id mol type x y z vx vy vz v_E_total v_Ke_total v_Pe_total", lengthunit="Angstrom", timeunit="ps")
    n_frames=sum(1 for e in u.trajectory[::step]) # step denotes the interval
    
    print("n_frames = {}".format(n_frames)) 
    
    for nf, frame in enumerate(u.trajectory[::step]):
        
        cl = freud.cluster.Cluster()       
        cl_props = freud.cluster.ClusterProperties()      
        dl = freud.cluster.Cluster()       
        dl_props = freud.cluster.ClusterProperties()  
        
        ts = mda.coordinates.base.Timestep(frame.positions[96288::,:].shape[0]) #The first 96288 atoms are carbon
        ts.positions=frame.positions[96288::,:] # Skip the first 96288 carbon atoms and pick the oxygen atoms, so that the cluster both on/not on the surface can be identified.
        ts.dimensions=frame.dimensions   
        
        cl.compute((frame.dimensions[:3], np.concatenate((frame.positions[:96288],frame.positions[96288::3]),axis=0)), neighbors={'r_max':3.36}) # the cutoff distance is 3.36 Å
        dl.compute((ts.dimensions[:3], ts.positions[::3]), neighbors={'r_max':3.36})    
             
        cl_props.compute((frame.dimensions[:3], np.concatenate((frame.positions[:96288],frame.positions[96288::3]),axis=0)), cl.cluster_idx)        
        dl_props.compute((ts.dimensions[:3], ts.positions[::3]), dl.cluster_idx)
              
        dlfa = pd.DataFrame(np.column_stack((dl.cluster_idx, np.arange(1,4455,1).reshape(4454,1))),columns=['cluster_idx', 'atom_idx']) # atomIndex of oxygen atoms in each water molecule, note that the atom index starts from 1.
        dfca = dlfa.groupby("cluster_idx") # get the oxygen atom index in the same cluster
        
        clusterIndex = []
        atomIndex = []
        for i in range(dl_props.centers.shape[0]):            
            if dl_props.centers[i] not in cl_props.centers: # find clusters on the surface 
                df_key = dfca.get_group(i) # get the atom index in cluster ith
                for a in df_key['atom_idx']:
                    clusterIndex.append(i)
                    atomIndex.append(a)
        
        np.savetxt("./{}.txt".format(nf),np.hstack([np.array(clusterIndex).reshape(-1,1), np.array(atomIndex).reshape(-1,1)]),delimiter=' ', fmt = '%s')

# step 2, growth rate
    for i in range(1, n_frames-1):     
        data = np.loadtxt('./{}.txt'.format(i)) # t0 
        data1 = np.loadtxt('./{}.txt'.format(i+1)) # t1
        
        df0 = pd.DataFrame(data,columns=['cluster_idx','atom_idx'])
        df_0 = df0.groupby("cluster_idx") # Finding Identical Clusters 
        df1 = pd.DataFrame(data1,columns=['cluster_idx','atom_idx'])
        df_1 = df1.groupby("cluster_idx") # Finding Identical Clusters
        
        cs_0 = [] # cluster size at t0
        d_s = []  # cluster growth from t0 to t1
        for key_0 in df_0.groups.keys():    # Key_0 of a cluster in the previous frame    
            df_key_0 = df_0.get_group(key_0)
            cluster_size_0 = df_key_0.size/2
            s_0 = df_key_0['atom_idx'] # atom index of the cluster of the previous frame
            max_common_size = 0
            common_size = 0
            max_key1 = key_0 # initialization
            flag_0 = 0
            for key_1 in df_1.groups.keys(): # Key_1 of a cluster in the latter frame 
                df_key_1 = df_1.get_group(key_1)
                s_1 = df_key_1['atom_idx'] #atom id of the cluster of the latter frame
                common_size = len(list(set(s_0).intersection(set(s_1)))) #common_size Number of clusters with a common atom index
                if common_size > max_common_size:
                    max_common_size = common_size
                    max_key1 = key_1
                    flag_0 = 1 # find a cluster has same atomidx both in the previous and latter frame
            if(flag_0):
                df_key_1_max = df_1.get_group(max_key1)
                s_1_max = df_key_1_max['atom_idx']
                flag_1 = 1
                for key_0 in df_0.groups.keys():
                    df_key_0 = df_0.get_group(key_0)
                    s_0 = df_key_0['atom_idx']           
                    common_size = len(list(set(s_1_max).intersection(set(s_0))))
                    if common_size > max_common_size:
                        flag_1 = 0
                        break
                if (flag_1):
                    delta_size = df_1.get_group(max_key1).size/2 - cluster_size_0
                    cs_0.append(cluster_size_0) 
                    d_s.append(delta_size)                
                    np.savetxt('./{}-{}.txt'.format(i,i+1), np.hstack([np.array(cs_0).reshape(-1,1), np.array(d_s).reshape(-1,1)]), delimiter=' ', newline='\n', header='', footer='', comments='#')
    
# step 3, compute the probability for cluster growth at different cluster sizes
    df = pd.DataFrame()
    for i in range(1, 14):#n_frames-1
        if(os.path.exists('./{}-{}.txt'.format(i, i+1))):
            temp_df = pd.read_csv('./{}-{}.txt'.format(i, i+1), sep=' ', names=['cluster_size', 'change'])# combine the cluster data at differet instances
            df = pd.concat([df, temp_df], axis=0).reset_index(drop=True)
        
    df = df.groupby("cluster_size") 
    A = np.zeros(len(df.groups.keys())) #  growth rate change from t0 to t0+delatt
    B = np.zeros(len(df.groups.keys())) #  decreae rate change from t0 to t0+delatt
    C = np.zeros(len(df.groups.keys())) #  cluster size at t0 

    for n, key in enumerate(df.groups.keys()):
        df_key = df.get_group(key)
        n_cluster = df_key.size/2
        C[n] = key        
        num1, num2 = np.unique(df_key['change'], return_counts=True)
        for i in range(len(num1)):
            if abs(num1[i]) < abs(key):
                if num1[i] > 0:
                    A[n] += (num1[i]*num2[i]/n_cluster)
                elif num1[i] == 0:
                    continue
                elif num1[i] < 0:
                    B[n] += (num1[i]*num2[i]/n_cluster)
    np.savetxt('./kinetic_analysis.txt', np.hstack([np.array(C).reshape(-1,1), np.array(A).reshape(-1,1), np.array(B).reshape(-1,1) ]), delimiter=' ', newline='\n', header='cluster_size, increase_rate, decrease_rate', footer='', comments='#')

step = 100
computeCondenseClusterSize('./dump.atom',step)