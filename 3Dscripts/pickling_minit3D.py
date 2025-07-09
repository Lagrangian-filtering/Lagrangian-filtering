import sys
import os
sys.path.append('../master_files/')
import pickle
import time 
import configparser
import json

from FileReaders import METHOD_HDF5
from MicroModels import IdealMHD_3D, timer_decorator
from Filters import smart_FindObs_root_parallel, smart_box_filter_parallel
from MesoModels import minitMHD_3D

if __name__ == '__main__':

    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.", flush=True)
        raise Exception()

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    # READING/SETTING UP THE MICROMODEL
    pickled_micro = True 

    if pickled_micro:
        pickle_where = config['Directories']['pickled_files_dir']
        pickle_name = config['Filenames']['micro_pickled_filename']
        with open(pickle_where + pickle_name, 'rb') as filehandle:
            micro_model = pickle.load(filehandle)

    else:
        hdf5_directory = config['Directories']['hdf5_dir']
        # filenames = hdf5_directory  
        snapshots_opts = json.loads(config['Micro_model_settings']['snapshots_opts'])
        fewer_snaps_required = snapshots_opts['fewer_snaps_required']
        smaller_list = snapshots_opts['smaller_list']
        FileReader = METHOD_HDF5(hdf5_directory, fewer_snaps_required, smaller_list)
        num_snaps = FileReader.num_files

        micro_model = IdealMHD_3D()
        read_in_data3D = timer_decorator(FileReader.read_in_data3D)
        read_in_data3D(micro_model)
        setup_structures = timer_decorator(micro_model.setup_structures)
        setup_structures()

        # SAVING THE MICRO DATA - pickling
        pickle_save = True
        pickle_where = config['Directories']['pickled_files_dir']
        pickle_name = config['Filenames']['micro_pickled_filename']

        if pickle_save:
            os.makedirs(pickle_where, exist_ok=True)
            with open(pickle_where + pickle_name, 'wb') as filehandle:
                pickle.dump(micro_model, filehandle)

    # SETTING UP THE MESO MODEL 
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
    find_obs = smart_FindObs_root_parallel(box_len)
    spatial_dims = micro_model.get_spatial_dims()
    filter = smart_box_filter_parallel(spatial_dims, width)

    meso_spatial_bdrs = [x_range, y_range, z_range]
    meso_model = minitMHD_3D(micro_model, find_obs, filter) 
    setup_mesogrid_smart = timer_decorator(meso_model.setup_mesogrid_smart)
    setup_mesogrid_smart(num_T_slices, meso_spatial_bdrs, coarse_factor = coarse_factor)


    # FINDING THE OBSERVERS AND FILTERING
    n_cpus = int(config['Meso_model_settings']['n_cpus'])
    random_select = bool(int(config['Meso_model_settings']['random_selection']))

    if random_select:
        n_random_points = int(config['Meso_model_settings']['random_selection'])
        find_observers_parallel_random = timer_decorator(meso_model.find_observers_parallel_random)
        find_observers_parallel_random(n_cpus, n_random_points)
    else:
        find_observers_parallel = timer_decorator(meso_model.find_observers_parallel)
        find_observers_parallel(n_cpus)
    
    filter_micro_vars_parallel= timer_decorator(meso_model.filter_micro_vars_parallel)
    filter_micro_vars_parallel(n_cpus)


    # SAVING THE MESO DATA - pickling
    pickle_save = True
    pickle_where = config['Directories']['pickled_files_dir']
    pickle_name = config['Filenames']['meso_pickled_filename']

    if pickle_save:
        os.makedirs(pickle_where, exist_ok=True)
        with open(pickle_where + pickle_name, 'wb') as filehandle:
            pickle.dump(meso_model, filehandle)



