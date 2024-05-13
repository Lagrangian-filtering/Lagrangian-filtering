import sys
# import os
sys.path.append('../../master_files/')
import pickle
import configparser
import json
import time
import math
from scipy import stats
from itertools import product
import multiprocessing as mp
from sklearn.metrics import mean_absolute_error

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *
from Analysis import *

if __name__ == '__main__':

    # #################################################################################
    # # GIVEN A VAR TO MODEL AND A LIST OF REGRESSORS, PERFORM THE REGRESSION 
    # # WITH ANY POSSIBLE COMBINATION OF REGRESSORS TAKEN FROM THE INPUT LIST
    # # FIND THE ONE THAT BEST DESCRIBE THE QUANITY TO BE MODELLED
    # ################################################################################# 
    
    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    # LOADING MESO MODEL
    pickle_directory = config['Directories']['pickled_files_dir']
    meso_pickled_filename = config['Filenames']['meso_pickled_filename']
    MesoModelLoadFile = pickle_directory + meso_pickled_filename

    print('================================================')
    print(f'Starting job on data from {MesoModelLoadFile}')
    print('================================================\n\n')

    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)

    # WHICH DATA YOU WANT TO RUN THE ROUTINE ON?
    dep_var_str = config['Find_best_fit_settings']['var_to_model']
    dep_var = meso_model.meso_vars[dep_var_str]
    regressors_strs = json.loads(config['Find_best_fit_settings']['regressors_strs'])
    regressors = []
    for i in range(len(regressors_strs)):
        temp = meso_model.meso_vars[regressors_strs[i]]
        regressors.append(temp)
    print(f'Dependent var: {dep_var_str},\n Explanatory vars: {regressors_strs}\n')

    
    # WHICH GRID-RANGES SHOULD WE CONSIDER?
    regression_ranges = json.loads(config['Find_best_fit_settings']['ranges'])
    x_range = regression_ranges['x_range']
    y_range = regression_ranges['y_range']
    num_slices_meso = int(config['Find_best_fit_settings']['num_T_slices'])
    time_of_central_slice = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)]
    ranges = [[time_of_central_slice, time_of_central_slice], x_range, y_range]

    # READING PREPROCESSING INFO FROM CONFIG FILE
    preprocess_data = json.loads(config['Find_best_fit_settings']['preprocess_data']) 
    add_intercept = not not int(config['Find_best_fit_settings']['add_intercept'])
    centralize = int(config['Find_best_fit_settings']['centralize'])
    test_percentage = float(config['Find_best_fit_settings']['test_percentage'])

    data = [dep_var]
    for i in range(len(regressors)):
        data.append(regressors[i])
    
    # PRE-PROCESSING: Trimming, pre-processing and splitting intro train + test set 
    statistical_tool = CoefficientsAnalysis() 
    model_points = meso_model.domain_vars['Points']
    new_data = statistical_tool.trim_dataset(data, ranges, model_points)
    new_data = statistical_tool.preprocess_data(new_data, preprocess_data)

    if centralize: 
        new_data, means = statistical_tool.centralize_dataset(new_data)
        dep_var_mean = means[0]
        regressors_means = means[1:]

    training_data, test_data = statistical_tool.split_train_test(new_data, test_percentage=test_percentage)

    dep_var_train = training_data[0]
    dep_var_test = test_data[0]
    regressors_train = training_data[1:]
    regressors_test = test_data[1:]


    # REGRESSING IN PARALLEL: consider all possible combination of input regressors' list
    # ALSO: EVALUATE GOODNESS OF FIT 
    num_regressors = len(regressors_train)
    bool_regressors_combs = [seq for seq in product((True, False), repeat=num_regressors)][0:-1]
    print(f'Number of tested combinations of regressors: {len(bool_regressors_combs)}\n')
    
    def parall_regress_task(comb_regressors):
        """
        """
        # DOING THE REGRESSION GIVEN SOME COMBINATION OF INPUT REGRESSORS
        actual_regressors = []
        actual_regressors_strs = []
        for i in range(num_regressors):
            if comb_regressors[i] == True:
                actual_regressors.append(regressors_train[i])
                actual_regressors_strs.append(regressors_strs[i])
        
        coeffs, _ = statistical_tool.scalar_regression(dep_var_train, actual_regressors, add_intercept=add_intercept)

        # BUILDING TEST-DATA PREDICTIONs GIVEN THE REGRESSED MODEL   
        actual_regressors = []
        for i in range(num_regressors):
            if comb_regressors[i] == True:
                actual_regressors.append(regressors_test[i])

        dep_var_model = np.zeros(dep_var_test.shape)
        if centralize:
            for i in range(len(actual_regressors)):
                dep_var_model += np.multiply(coeffs[i], actual_regressors[i]) 

        else:
            if add_intercept:
                dep_var_model += coeffs[0]
                for i in range(len(actual_regressors)):
                    dep_var_model += np.multiply(coeffs[i+1], actual_regressors[i]) 
            elif not add_intercept: 
                for i in range(len(actual_regressors)):
                    dep_var_model += np.multiply(coeffs[i], actual_regressors[i]) 
   
        # r, _ = stats.pearsonr(dep_var_test, dep_var_model)
        # return r, coeffs , comb_regressors
    
        # mean_error = mean_absolute_error(dep_var_test, dep_var_model)
        # return mean_error, coeffs , comb_regressors

        w = statistical_tool.wasserstein_distance(dep_var_test, dep_var_model, sample_points=300)            
        return w, coeffs, comb_regressors
                    
    
    n_cpus = int(config['Find_best_fit_settings']['n_cpus'])
    # pearsons = []
    # mean_errors = []
    wassersteins = []
    fitted_coeffs = []
    regressors_combinations = []
    with mp.Pool(processes=n_cpus) as pool:
        print('Performing all possible regression of {} in parallel with {} processes\n'.format(dep_var_str, pool._processes), flush=True)
        for result in pool.map(parall_regress_task, bool_regressors_combs): 
            # pearsons.append(result[0])
            # mean_errors.append(result[0])
            wassersteins.append(result[0])
            fitted_coeffs.append(result[1])
            regressors_combinations.append(result[2])


    # FINDING BEST MODEL, RE-BUILDING DATA PREDICITON FOR TEST SET
    # rs_squared = np.power(pearsons, 2)
    # max_r_index = np.argmax(rs_squared)
    # best_coeffs = fitted_coeffs[max_r_index]
    # best_regressors_combination = regressors_combinations[max_r_index]
    # ordering_idx = np.argsort(rs_squared)
    # print('Printing max r-squared and the corresponding regression combination:')

    min_w_index = np.argmin(wassersteins)
    best_coeffs = fitted_coeffs[min_w_index]
    ordering_idx = np.argsort(wassersteins)
    

    # for i in range(-1,-6,-1):
    for i in range(0,6,1):
        index = ordering_idx[i]
        which_regressors = []
        for j in range(len(regressors_combinations[index])):
            if regressors_combinations[index][j] == True:
                which_regressors.append(regressors_strs[j])
        # print(f'r^2: {rs_squared[index]}, regressors: {which_regressors}')
        print(f'wasserstein: {wassersteins[index]}, regressors: {which_regressors}')

    print(f'\nBest coefficients: {best_coeffs}')

    