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

    #######################################################
    # SCRIPT TO TEST UPDATES TO MICRO-MODEL ROUTINES
    #######################################################

    # READING SIMULATION SETTINGS FROM CONFIG FILE

    config = configparser.ConfigParser()
    config.read(sys.argv[1])


    hdf5_directory = config['Directories']['hdf5_dir']
    print('=========================================================================')
    print(f'Starting job on data from {hdf5_directory}')
    print('=========================================================================\n\n')

    snapshots_opts = json.loads(config['Micro_model_settings']['snapshots_opts'])
    fewer_snaps_required = snapshots_opts['fewer_snaps_required']
    smaller_list = snapshots_opts['smaller_list']
    
    start_time = time.perf_counter()
    FileReader = METHOD_HDF5(hdf5_directory, fewer_snaps_required, smaller_list)
    time_taken = time.perf_counter() - start_time
    print('Time taken to initialize File Reader: {}\n'.format(time_taken))

    micro_bis = IdealHD_3D()

    start_time = time.perf_counter()
    FileReader.read_in_data3D(micro_bis)
    time_taken = time.perf_counter() - start_time
    print('Time taken to read in data: {}\n'.format(time_taken))

    n_cpus = int(config['Meso_model_settings']['n_cpus'])
    start_time = time.perf_counter()
    micro_bis.setup_structures_parallel(n_cpus)
    time_taken_parallel = time.perf_counter() - start_time

    print('Time taken to setup micro structures in parallel: {}'.format(time_taken_parallel))
    # print('Finished reading micro data from hdf5, structures also set up.')


    micro_model = IdealHD_3D()

    start_time = time.perf_counter()
    FileReader.read_in_data3D(micro_model)
    time_taken = time.perf_counter() - start_time
    print('Time taken to read in data: {}\n'.format(time_taken))

    start_time = time.perf_counter()
    micro_model.setup_structures()
    time_taken_serial = time.perf_counter() - start_time
    speed_up_factor = time_taken_serial / time_taken_parallel
    print('Time taken to setup micro structures in serial: {}\nSpeed up factor: {}'.format(time_taken_serial, speed_up_factor))
