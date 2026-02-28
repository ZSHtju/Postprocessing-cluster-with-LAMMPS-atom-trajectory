This repository contains LAMMPS and Python scripts used in our publication in Droplet (DOI: 10.1002/dro2.70064).

The simulation outputs dump files every 1000 timesteps. Since the Grand Canonical Monte Carlo (GCMC) method is employed, water molecules are added or removed every 500 steps to maintain a constant chemical potential, causing the total number of molecules to vary over time. However, MDAnalysis cannot directly process dump files with a dynamically changing number of molecules. To address this, we first use the split_dump_data.py script to split the dump file into individual timestep files.

The following scripts are then used for analysis:

analyze_condensed_clusters.py: 
Identifies condensed clusters on the graphite surface at each timestep using the Freud library, and classifies their locations into three categories: inside pores (1), on the surface but outside pores (0), or in the air (2). Outputs cluster properties including size, center coordinates, and position type.

map_cluster_water_relationship.py: 
Establishes the mapping relationship between water molecule IDs and cluster IDs.

trace_back_cluster.py
Step 1: Implements a backtracking method to trace each cluster that ultimately exceeds the critical nucleation size, recording its center coordinates and molecular count at every timestep.
Step 2: Classifies the evolutionary pathways of clusters into four distinct modes and calculates the condensation growth rate for each mode.