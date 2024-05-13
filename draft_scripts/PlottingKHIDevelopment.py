# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:03:27 2024

@author: marcu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:38:23 2024

@author: marcu
"""

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import glob
import os
import pickle
#from os import system
import sys as system

if __name__ == "__main__":

    data_dir = '/mainfs/filtering/TC_data_draft/800X800/'
    subfolders = ['ET02', 'ET04', 'ET06', 'ET08', 'ET12', 'ET16']
    HDF5_dir = '/METHOD_output/'
    filenames = ['ET_02_00.hdf5', 'ET_04_00.hdf5','ET_06_00.hdf5', 'ET_08_00.hdf5','ET_12_00.hdf5','ET_16_00.hdf5']
    HDF5_files = []
    for subfolder, filename in zip(subfolders, filenames):
        HDF5_files.append(data_dir+subfolder+HDF5_dir+filename)

    fss = []
    for HDF5_file in HDF5_files:
        fss.append(h5py.File(HDF5_file, 'r'))

    data_to_plot = []

    for fs in fss:
        data_to_plot.append(fs['Primitive/rho'])

    t_labels = [r'$t=2.0$',r'$t=4.0$',r'$t=6.0$',r'$t=8.0$',r'$t=12.0$',r'$t=16.0$']

    fig, axes = plt.subplots(2,3,figsize=(12,8), sharex=True, sharey=True)#, constrained_layout=True)
    for ax, data, i in zip(axes.flatten(), data_to_plot, range(len(data_to_plot))):
        im = ax.imshow(data, extent=[-1.0,1.0,-1.0,1.0])
        if i == 1 or i == 2 or i == 4 or i == 5:
            ax.tick_params(axis='y',left=False)
        if i == 0 or i == 1 or i == 2:
            ax.tick_params(axis='x',bottom=False)
        if i == 3 or i == 5:
            ax.set_xticks([-0.75,-0.25,0.25,0.75])
        if i == 4:
            ax.set_xticks([-1.0,-0.5,0.0,0.5,1.0])
        #    ax.set_yticks([])
        ax.set_title(t_labels[i])

    #axes[0,1].set_xticks([])

    fig.subplots_adjust(right=0.8,wspace=0.01, hspace=0.0001)
    cbar_ax = fig.add_axes([0.815, 0.15, 0.035, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig('6plot_testing.pdf', bbox_inches='tight',dpi=1200)





























