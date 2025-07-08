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

    ####################################################################################################
    # SCRIPT TO FILTER-AND-VISUALIZE MESO-MODEL QUANTITIES: mainly for testing routines
    ####################################################################################################

    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])


    hdf5_directory = config['Directories']['hdf5_dir']
    print('=========================================================================')
    print(f'Starting job on data from {hdf5_directory}')
    print('=========================================================================\n\n')

    snapshots_opts = json.loads(config['Micro_model_settings']['snapshots_opts'])
    fewer_snaps_required = snapshots_opts['fewer_snaps_required']
    smaller_list = snapshots_opts['smaller_list']
    FileReader = METHOD_HDF5(hdf5_directory, fewer_snaps_required, smaller_list)
    num_snaps = FileReader.num_files
    micro_model = IdealHD_3D()
    FileReader.read_in_data3D(micro_model)
    micro_model.setup_structures()
    print('Finished reading micro data from hdf5, structures also set up.')
    print('Micro-model times: {}'.format(micro_model.domain_vars['t']))
    print('Micro_model dx: {}'.format(micro_model.domain_vars['dx']))

    # SETTING UP THE MESO MODEL 
    start_time = time.perf_counter()
    meso_grid = json.loads(config['Meso_model_settings']['meso_grid_smart'])
    filtering_options = json.loads(config['Meso_model_settings']['filtering_options'])

    coarse_factor = meso_grid['coarse_grain_factor']
    num_T_slices = int(meso_grid['num_T_slices'])

    x_range = meso_grid['x_range']
    y_range = meso_grid['y_range']
    z_range = meso_grid['z_range']

    box_len_ratio = float(filtering_options['box_len_ratio'])
    filter_width_ratio =  float(filtering_options['filter_width_ratio'])
    box_len = box_len_ratio * micro_model.domain_vars['dx']
    width = filter_width_ratio * micro_model.domain_vars['dx']
    find_obs = FindObs_root_parallel(micro_model, box_len)
    filter = box_filter_parallel(micro_model, width)

    meso_spatial_bdrs = [x_range, y_range, z_range]
    meso_model = resHD_3D(micro_model, find_obs, filter) 
    meso_model.setup_mesogrid_smart(num_T_slices, meso_spatial_bdrs, coarse_factor = coarse_factor)
    print('Finished setting up the meso_grid.')

    num_points = meso_model.domain_vars['Nt'] * meso_model.domain_vars['Nx'] * meso_model.domain_vars['Ny'] * meso_model.domain_vars['Nz']
    Xmin = meso_model.domain_vars['Xmin']
    Xmax = meso_model.domain_vars['Xmax']
    Ymin = meso_model.domain_vars['Ymin']
    Ymax = meso_model.domain_vars['Ymax']
    Zmin = meso_model.domain_vars['Zmin']
    Zmax = meso_model.domain_vars['Zmax']

    print('Time coord of meso_slices: {}'.format(meso_model.domain_vars['T']), flush=True)
    print('Meso_model grid spacing: {}\nMeso_model time gap between slices: {}\n'.format(meso_model.domain_vars['Dx'], meso_model.domain_vars['Dt']))
    print('Xmin, Xmax: {}-{}\nYmin, Ymax: {}-{}\nZmin, Zmax: {}-{}\n'.format(Xmin, Xmax, Ymin, Ymax, Zmin, Zmax))
    print('Tot num points: {}\n'.format(num_points), flush=True)

    # # ###################################################### 
    # # # SETTING UP THE MESO MODEL USING setup_meso_grid()
    # # ######################################################
    # # start_time = time.perf_counter()
    # # meso_grid = json.loads(config['Meso_model_settings']['meso_grid'])
    # # filtering_options = json.loads(config['Meso_model_settings']['filtering_options'])

    # # coarse_factor = meso_grid['coarse_grain_factor']
    # # coarse_time = meso_grid['coarse_grain_time']

    # # t_range = meso_grid['t_range']
    # # x_range = meso_grid['x_range']
    # # y_range = meso_grid['y_range']
    # # z_range = meso_grid['z_range']

    # # box_len_ratio = float(filtering_options['box_len_ratio'])
    # # filter_width_ratio =  float(filtering_options['filter_width_ratio'])

    # # box_len = box_len_ratio * micro_model.domain_vars['dx']
    # # width = filter_width_ratio * micro_model.domain_vars['dx']
    # # find_obs = FindObs_root_parallel(micro_model, box_len)
    # # filter = box_filter_parallel(micro_model, width)
    # # meso_model = resHD_3D(micro_model, find_obs, filter) 
    # # meso_model.setup_meso_grid([t_range, x_range, y_range, z_range], coarse_factor = coarse_factor, coarse_time = coarse_time)
    # # print('Finished setting up the meso_grid.')
    # # num_points = meso_model.domain_vars['Nt'] * meso_model.domain_vars['Nx'] * meso_model.domain_vars['Ny'] * meso_model.domain_vars['Nz']
    # # print('Num meso_slices: {}'.format(meso_model.domain_vars['Nt']), flush=True)
    # # print('Tot num points: {}\n'.format(num_points), flush=True)
    # # ######################################################
    
    
    # ######################################################
    # # FINDING THE OBSERVERS ETCETERA
    # ######################################################
    n_cpus = int(config['Meso_model_settings']['n_cpus'])

    start_time = time.perf_counter()
    meso_model.find_observers_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Observers found in parallel: time taken= {}\n'.format(time_taken), flush=True)

    # FILTERING
    start_time = time.perf_counter()
    meso_model.filter_micro_vars_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Parallel filtering stage ended (fw= {}): time taken= {}\n'.format(int(filter_width_ratio), time_taken), flush=True)

    # DECOMPOSING AND CALCULATING THE CLOSURE INGREDIENTS
    start_time = time.perf_counter()
    meso_model.decompose_structures_parallel(n_cpus)
    time_taken = time.perf_counter() - start_time
    print('Finished decomposing meso structures in parallel, time taken: {}\n'.format(time_taken), flush=True)

    slices = list(json.loads(config['Meso_model_settings']['slices_T_derivs']))
    if len(slices) ==0: 
        slices=None
    start_time = time.perf_counter()
    meso_model.calculate_derivatives(slices=slices)
    time_taken = time.perf_counter() - start_time
    print('Finished computing derivatives (serial), time taken: {}\n'.format(time_taken), flush=True)

    saving_directory = config['Directories']['pickled_files_dir']
    meso_pickled_filename = config['Directories']['meso_pickled_filename']
    with open(saving_directory + meso_pickled_filename, 'wb') as filehandle: 
        pickle.dump(meso_model, filehandle)

