import sys
# import os
sys.path.append('../master_files/')
import configparser
import json
import pickle

from FileReaders import *
from MicroModels import *
from Visualization import *

if __name__ == '__main__':

    #############################################################################
    # SCRIPT TO PLOT MICRO-MODEL QUANTITIES: DATA DIRECTLY FROM SIMULATIONS
    #############################################################################

    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    # LOADING MICRO DATA FROM HDF5 OR PICKLE
    from_hdf5 = True 
    meso_or_micro = False

    if from_hdf5:
        hdf5_directory = config['Directories']['hdf5_dir']
        print('=========================================================================')
        print(f'Starting job on data from {hdf5_directory}')
        print('=========================================================================\n\n')
        filenames = hdf5_directory 
        snapshots_opts = json.loads(config['Micro_model_settings']['snapshots_opts'])
        fewer_snaps_required = snapshots_opts['fewer_snaps_required']
        smaller_list = snapshots_opts['smaller_list']
        FileReader = METHOD_HDF5(filenames, fewer_snaps_required, smaller_list)
        num_snaps = FileReader.num_files
        micro_model = IdealHD_3D()
        FileReader.read_in_data3D(micro_model)
        micro_model.setup_structures()
        print('Finished reading micro data from hdf5, structures also set up.')

    else: 
        pickle_directory = config['Directories']['pickled_files_dir']
        pickled_filename = config['Directories']['pickled_filename']
        ModelLoadFile = pickle_directory + pickled_filename
        print('=========================================================================')
        print(f'Starting job on data from {ModelLoadFile}')
        print('=========================================================================\n\n')

        with open(ModelLoadFile, 'rb') as filehandle: 
            if meso_or_micro: 
                meso_model = pickle.load(filehandle)
                micro_model = meso_model.micro_model
            else:
                micro_model = pickle.load(filehandle)

    print('micro_t: {}'.format(micro_model.domain_vars['t']))

    # PLOT SETTINGS
    plot_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    plot_time = plot_ranges['t']
    slicing_axis = int(plot_ranges['slicing_axis'])
    slicing_const = float(plot_ranges['slicing_const'])
    range_1 = plot_ranges['range_1']
    range_2 = plot_ranges['range_2']

    save_plot = json.loads(config['Plot_settings']['save_plot'])
    saving_directory = config['Directories']['figures_dir']
    visualizer = Plotter_2Dslices([11.97, 8.36])

    # FINALLY, PLOTTING
    # Plotting Primitives+
    if meso_or_micro: 
        model = meso_model
    else: 
        model = micro_model
    vars = ['W', 'vx', 'vy', 'vz', 'n', 'p']
    norms= None #['log', 'symlog', 'symlog', 'symlog', 'log', 'log']
    cmaps = None # ['plasma', 'plasma', 'plasma', 'plasma']
    components = None #[(0,), (1,), (2,), (3,)]
    fig=visualizer.plot_vars(model, vars, plot_time, slicing_axis, slicing_const, range_1, range_2, components_indices = components, 
                            method='raw_data', norms=norms, cmaps=cmaps)
    fig.tight_layout()

    if save_plot:
        time_for_filename = str(round(plot_time,2))
        filename = "/micro_t" + time_for_filename + "_prims.png"
        plt.savefig(saving_directory + filename, format = "png", dpi=300)
    else: 
        plt.show()

    