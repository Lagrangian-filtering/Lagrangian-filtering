# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:38:23 2024

@author: Marcus
"""

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import h5py
import glob
import os
import pickle
import sys
sys.path.append('../master_files/')
from system.BaseFunctionality import *
from MicroModels import * 
from Analysis import *
from MesoModels import * 

if __name__ == "__main__":
    
    # Load models' pickle files
    hr_pickle_dirs = ['./Output/t_998_1002/800x1600/Pickles/', './Output/t_1998_2002/800x1600/Pickles/', './Output/t_2998_3002/800x1600/Pickles/']
    lr_pickle_dirs = ['./Output/t_998_1002/400x800/Pickles/', './Output/t_1998_2002/400x800/Pickles/', './Output/t_2998_3002/400x800/Pickles/']
    HRKESpecs = []
    LRKESpecs = []
    ts = [10.0,20.0,30.0]
    Nxs = []

    for pickle_dir in hr_pickle_dirs:
        with open(pickle_dir+'IdealHydro2D.pickle', 'rb') as filehandle:
            model = pickle.load(filehandle)
        # Dump their spectra
        KESpec = 0
        KESpec = Base.getPowerSpectrumSq(model, FourierAnalysis.GetKESF(model))

        with open(pickle_dir+'KESpec.pickle', 'wb') as filehandle:
            pickle.dump(KESpec, filehandle)

        HRKESpecs.append(KESpec)
    Nxs.append(model.domain_vars['nx']//2)

    for pickle_dir in lr_pickle_dirs:
        with open(pickle_dir+'IdealHydro2D.pickle', 'rb') as filehandle:
            model = pickle.load(filehandle)

        KESpec = 0
        KESpec = Base.getPowerSpectrumSq(model, FourierAnalysis.GetKESF(model))

        with open(pickle_dir+'KESpec.pickle', 'wb') as filehandle:
            pickle.dump(KESpec, filehandle)

        LRKESpecs.append(KESpec)
    Nxs.append(model.domain_vars['nx']//2)
    
    # Plot the spectra
    fig, axs = plt.subplots(1, 3, figsize=(8.4,2.8), sharey=True, dpi=1200)
    for ax, HRKESpec, LRKESpec, t in zip(axs, HRKESpecs, LRKESpecs, ts):
        ax.loglog(np.arange(1, Nxs[0]+1), (np.arange(1, Nxs[0]+1)**(5/3))*np.arange(1, Nxs[0]+1)*HRKESpec)
        ax.loglog(np.arange(1, Nxs[1]+1), (np.arange(1, Nxs[1]+1)**(5/3))*np.arange(1, Nxs[1]+1)*LRKESpec)
        ax.set_xlabel(r'$k$')
        ax.set_title(r'$t =~$'+str(t))

    axs[0].set_ylabel(r"$k^{8/3}|P_{T}(k)|^2$", {'fontsize':'large'})
    axs[1].set_yticks([])
    axs[2].set_yticks([])

    fig.tight_layout()
    plt.savefig('./KESpecs_IHD_t=10_20_30.pdf')
    plt.close()































