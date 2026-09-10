import sys
sys.path.append('../master_files/')
import configparser
import json
import pickle
import time

from FileReaders import *
from MicroModels import *
from Filters import *
from MesoModels import *

if __name__ == '__main__':

    ####################################################################################################
    # SCRIPT TO FILTER 3D IDEAL-MHD DATA AND FIT THE GENERIC OHM'S LAW CLOSURE, THEN PICKLE THE RESULT
    ####################################################################################################

    if len(sys.argv) == 1:
        print("You must pass the configuration file for the simulations.")
        raise Exception()

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    hdf5_directory = config['Directories']['hdf5_dir']
    print('=========================================================================')
    print(f'Starting MHD filtering job on data from {hdf5_directory}')
    print('=========================================================================\n\n')

    snapshots_opts = json.loads(config['Micro_model_settings']['snapshots_opts'])
    FileReader = METHOD_HDF5(hdf5_directory, snapshots_opts['fewer_snaps_required'], snapshots_opts['smaller_list'])

    micro_model = IdealMHD_3D()
    FileReader.read_in_data3D(micro_model)
    micro_model.setup_structures()
    print('Finished reading micro data from hdf5, structures also set up.', flush=True)
    print('Micro-model times: {}'.format(micro_model.domain_vars['t']))

    meso_grid = json.loads(config['Meso_model_settings']['meso_grid_smart'])
    filtering_options = json.loads(config['Meso_model_settings']['filtering_options'])

    coarse_factor = meso_grid['coarse_grain_factor']
    num_T_slices = int(meso_grid['num_T_slices'])
    meso_spatial_bdrs = [meso_grid['x_range'], meso_grid['y_range'], meso_grid['z_range']]

    box_len = float(filtering_options['box_len_ratio']) * micro_model.domain_vars['dx']
    width = float(filtering_options['filter_width_ratio']) * micro_model.domain_vars['dx']
    find_obs = FindObs_root_parallel(micro_model, box_len)
    filter = box_filter_parallel(micro_model, width)

    meso_model = resMHD_3D(micro_model, find_obs, filter)
    meso_model.setup_mesogrid_smart(num_T_slices, meso_spatial_bdrs, coarse_factor)
    print('Finished setting up the meso_grid.', flush=True)

    n_cpus = int(config['Meso_model_settings']['n_cpus'])

    start_time = time.perf_counter()
    meso_model.find_observers_parallel(n_cpus)
    print('Observers found: time taken= {}\n'.format(time.perf_counter() - start_time), flush=True)

    start_time = time.perf_counter()
    meso_model.filter_micro_vars_parallel(n_cpus)
    print('Filtering ended: time taken= {}\n'.format(time.perf_counter() - start_time), flush=True)

    start_time = time.perf_counter()
    meso_model.decompose_structures_parallel(n_cpus)
    meso_model.decompose_EM_parallel(n_cpus)
    print('Decomposition ended: time taken= {}\n'.format(time.perf_counter() - start_time), flush=True)

    ohms_law_terms = json.loads(config['Ohms_law_settings']['terms'])
    start_time = time.perf_counter()
    meso_model.fit_ohms_law_closure(
        use_resistive=ohms_law_terms['resistive'],
        use_dynamo=ohms_law_terms['dynamo'],
        use_hall=ohms_law_terms['hall'])
    print('Ohms law closure fit ended: time taken= {}\n'.format(time.perf_counter() - start_time), flush=True)

    saving_directory = config['Directories']['pickled_files_dir']
    meso_pickled_filename = config['Directories']['meso_pickled_filename']
    with open(saving_directory + meso_pickled_filename, 'wb') as filehandle:
        pickle.dump(meso_model, filehandle)
