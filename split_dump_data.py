'''
Script for splitting a large LAMMPS dump file into individual step files.

Input:./dump.atom (LAMMPS trajectory file containing multiple timesteps)
Output:./dumpdata/step_{timestep}.txt (individual dump files for each saved frame)

software: Pandas, NumPy
Author: Shenghui Zhong, Boxuan Cui (Beihang University)
'''

import pandas as pd
import numpy as np
import os

class dump:
    def __init__(self, time_step, xyz_df):
        self.time_step = time_step
        self.len = len(xyz_df)
        self.xyz_df = xyz_df

def read_dumps(file_path, extra_line_num, step_interval):
    
    num_of_atom = int(pd.read_csv(file_path, skiprows=3, nrows=1, header=None)[0])#Read the number of atoms, the 4th line contains the atom count.

    with open(file_path, 'r') as f:
        lines = f.readlines()
        box_bounds = [line.strip() for line in lines[5:8]] #Lines 6-8 contain the box dimensions, read the box dimensions.
    box_bounds = [f"{int(float(bound.split()[0]))} {int(float(bound.split()[1]))}" for bound in box_bounds]
    print(box_bounds)
    dump_list = []
    line_pointer = 0
    while True:
        try:
            time_step_df = pd.read_csv(file_path, skiprows=line_pointer + 1, nrows=1, header=None) #Read the current timestep, skip 1 line.
        except:
            break
        time_step = int(time_step_df[0])
        print(time_step)
        xyz_df = pd.read_csv(file_path, skiprows=line_pointer + 9, nrows=num_of_atom, sep=' ', header=None)  #Read the atomic coordinate matrix
        xyz_df = xyz_df.iloc[:,:]
        #print(xyz_df) 
        if not os.path.exists('./dumpdata'):
            os.makedirs('./dumpdata')
        if time_step % step_interval == 0:
            with open("./dumpdata/step_{}.txt".format(time_step), 'w') as f:
                f.write("ITEM: TIMESTEP\n")  #Write the header for subsequent reading with the MDAnalysis library.
                f.write("{}\n".format(time_step))  #n_step % 100 == 0 Read atomic coordinates every 100 steps.
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write("{}\n".format(num_of_atom))
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                for bound in box_bounds:
                    f.write("{}\n".format(bound))

                f.write("ITEM: ATOMS id mol type x y z vx vy vz\n")
                np.savetxt(f, xyz_df, fmt=['%d', '%d', '%d', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f', '%.6f'])
                print('save 1 step')
        
        line_pointer += (extra_line_num + num_of_atom)
        num_of_atom_df = pd.read_csv(file_path, skiprows=line_pointer + 3, nrows=1, header=None)#Read the atom count at the next timestep.
        num_of_atom = int(num_of_atom_df[0])
        print('Time Step: ', time_step, ' completed.')
        print('num_of_atom:', num_of_atom)
    return dump_list


dump_list = read_dumps('./dump.atom', 9, 10000)
