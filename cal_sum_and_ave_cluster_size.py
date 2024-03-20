'''
Calculate the properties of condensed clusters on a graphite.
calculate the condensation nucleation rates using the Yasuoka-Matsumoto (YM) method, Kenji Yasuoka and Mitsuhiro Matsumoto, Journal of Chemical Physics, 1998, 109(19):8451-8462.
software: MDAnalysis, Freud
file: LAMMPS atom trajectory
Author: Shenghui Zhong, Zheyu Shi (Beihang University)
'''

import os
import MDAnalysis as mda
import freud
import numpy as np
from pathlib import Path

def computeCondenseClusterSize(filename,step):   
    print('hello!!!')   
    
    u = mda.Universe(filename, format="LAMMPSDUMP", atom_style="id mol type x y z vx vy vz", lengthunit="Angstrom", timeunit="ps") # atom_style follows the order in the dump file. Usually, we need the positions.   
    n_frames=sum(1 for e in u.trajectory[::step])
    
    print("n_frames = {}".format(n_frames))  
    
    n2 = np.zeros(n_frames)
    sum_size = np.zeros(n_frames)
    ave_size = np.zeros(n_frames)
    
    for nf, frame in enumerate(u.trajectory[::step]):   
        cl = freud.cluster.Cluster()       
        cl_props = freud.cluster.ClusterProperties()
        
        dl = freud.cluster.Cluster()       
        dl_props = freud.cluster.ClusterProperties()
        
        ts = mda.coordinates.base.Timestep(frame.positions[96288::,:].shape[0]) #The first 96288 atoms are carbon
        ts.positions=frame.positions[96288::,:] # Skip the first 96288 carbon atoms and pick the oxygen atoms, so that the cluster both on/not on the surface can be identified.
        ts.dimensions=frame.dimensions
        
        cl.compute((frame.dimensions[:3], np.concatenate((frame.positions[:96288],frame.positions[96288::3]),axis=0)), 
        neighbors={'r_max':3.36}) # pick all atoms, so that the water cluster on the surface will be identified as one big cluster, the cutoff distance is 3.36 Å
        
        dl.compute((ts.dimensions[:3], ts.positions[::3]), neighbors={'r_max':3.36}) # the cutoff distance is 3.36 Å
               
        cl_props.compute((frame.dimensions[:3], np.concatenate((frame.positions[:96288],frame.positions[96288::3]),axis=0)), cl.cluster_idx) 
       
        dl_props.compute((ts.dimensions[:3], ts.positions[::3]), dl.cluster_idx)
        
        el_props_centers = []
        el_props_sizes = []
     
        for i in range(dl_props.centers.shape[0]):            
            if dl_props.centers[i] not in cl_props.centers:  # Here the clusters on the carbon surface are calculated
                el_props_centers.append(dl_props.centers[i])
                el_props_sizes.append(dl_props.sizes[i])
                
        n_cluster_arrays = {}
        for c_size in range(2,15):
            key = f"n_{c_size}"
            n_cluster_arrays[key] = np.zeros(n_frames)
            
        for i in range(len(el_props_sizes)):           
            if el_props_sizes[i] >= 2: # Calculate the total number of water molecules condensed on the carbon surface
                sum_size[nf] += el_props_sizes[i]
                n2[nf] += 1
            for c_size in range(2,15): # for YM method, the range(2, 15) should be adjusted up to specific case.
                if el_props_sizes[i] >= c_size:
                    n_cluster_arrays["n_{}".format(c_size)] += 1
        if n2[nf] != 0:  #Calculate the average cluster size
            ave_size[nf] = sum_size[nf]/n2[nf]  
    for c_size in range(2,15):        
        np.save('./n_cluster_{}.npy'.format(c_size), n_cluster_arrays["n_{}".format(c_size)])
         
    return  ave_size, sum_size

step = 100 
#The folder where the trajectory files are located, modified according to the path   
# Start reading trajectory files
ave_cluster_on_graphite, sum_cluster_on_graphite = computeCondenseClusterSize('./dump.atom', step)
np.save('./ave_cluster_on_graphite.npy', ave_cluster_on_graphite)
np.save('./sum_cluster_on_graphite.npy', sum_cluster_on_graphite)
