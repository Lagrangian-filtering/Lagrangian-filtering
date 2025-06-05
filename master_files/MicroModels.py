# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:00:02 2023

@authors: Thomas & Marcus 
"""

import numpy as np
import math
import time
import os
import multiprocessing as mp
from itertools import product
from time import perf_counter

from scipy.interpolate import interpn 
from multimethod import multimethod

from FileReaders import *
from system.BaseFunctionality import *


# These are the symbols, so be careful when using these to construct vectors!
# levi3D = np.array([[[ np.sign(i-j) * np.sign(j- k) * np.sign(k-i) \
#                       for k in range(3)]for j in range(3) ] for i in range(3) ])

levi4D = np.array([[[[ np.sign(i - j) * np.sign(j - k) * np.sign(k - l) * np.sign(i - l) \
                       for l in range(4)] for k in range(4) ] for j in range(4)] for i in range(4)])

def timer_decorator(func): 
    """
    Timing function decorator. 

    Print to output the time taken to execute the function. 
    """
    def wrap_func(*args, **kwargs): 
        t1 = perf_counter() 
        result = func(*args, **kwargs) 
        t2 = perf_counter() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func 


class IdealHD_2D(object):
    """
    Class for storing data of 2D hydrodynamic simulations (of turbulence). 
    """
    def __init__(self, interp_method = "linear"):
        """
        Sets up the main variables and dictionaries. Var and dict names need not match
        those of the output data, this can be taken into account using 
        classes in FileReaders.py 

        Parameters
        ----------
        interp_method: str
            optional method to be used by interpn        
        """
        self.spatial_dims = 2
        self.interp_method = interp_method

        self.metric = np.zeros((3,3))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = +1

        #Dictionary for grid: info and points
        self.domain_int_strs = ('nt','nx','ny')
        self.domain_float_strs = ("tmin","tmax","xmin","xmax","ymin","ymax","dt","dx","dy")
        self.domain_array_strs = ("t","x","y","points")
        self.domain_vars = dict.fromkeys(self.domain_int_strs+self.domain_float_strs+self.domain_array_strs)
        for str in self.domain_vars:
            self.domain_vars[str] = []   

        #Dictionary for primitive var
        self.prim_strs = ("vx","vy","n","p")
        self.prim_vars = dict.fromkeys(self.prim_strs)
        for str in self.prim_strs:
            self.prim_vars[str] = []

        #Dictionary for auxiliary var
        self.aux_strs = ("W", "h", "e")
        self.aux_vars = dict.fromkeys(self.aux_strs)
        for str in self.aux_strs:
            self.aux_vars[str] = []

        #Dictionary for structures
        self.structures_strs = ("BC", "SET", "bar_vel")
        self.structures = dict.fromkeys(self.structures_strs)
        for str in self.structures_strs:
            self.structures[str] = []

        self.labels_var_dict = {'BC' : r'$n^{a}$', 
                                'SET' : r'$T^{ab}$', 
                                'bar_vel' : r'$u^a$',
                                'vx' : r'$v_x$', 
                                'vy' : r'$v_y$', 
                                'n' : r'$n$', 
                                'W' : r'$W$', 
                                'e' : r'$e$', 
                                'h' : r'$h$', 
                                'p' : r'$p$'} 

    def upgrade_labels_dict(self, entry_dict):
        """
        Add/change dictionry key/value entry for figure labels.
        """
        self.labels_var_dict.update(entry_dict)
        
    def get_spatial_dims(self):
        return self.spatial_dims

    def get_model_name(self):
        return 'Ideal Hydro (2+1d)'

    def get_domain_strs(self):
        return self.domain_int_strs + self.domain_float_strs + self.domain_array_strs
    
    def get_prim_strs(self):
        return self.prim_strs
    
    def get_aux_strs(self):
        return self.aux_strs
    
    def get_structures_strs(self):
        return self.structures_strs
    
    def get_all_var_strs(self):
        return self.get_prim_strs() + self.get_aux_strs() + self.get_structures_strs()
    
    def get_gridpoints(self): 
        return self.domain_vars['points']

    def get_interpol_var(self, var, point):
        """
        Returns quantity corresponding to input str 'var' evaluated 
        at the input point 'point' via interpolation.

        Parameters
        ----------
        var: str corresponding to primitive, auxiliary or structre variable
            
        point : list of floats
            ordered coordinates: t,x,y

        Return
        ------
        Interpolated values/arrays corresponding to variable. 
        Empty list if none of the variables is a primitive, auxiliary o structure of the micro_model

        Notes
        -----
        Interpolation may fail when too close to boundary
        """

        if var in self.get_prim_strs():
            return interpn(self.domain_vars['points'], self.prim_vars[var], point, method = self.interp_method)[0]
        elif var in self.get_aux_strs():
            return interpn(self.domain_vars['points'], self.aux_vars[var], point, method = self.interp_method)[0]
        elif var in self.get_structures_strs():
            return interpn(self.domain_vars['points'], self.structures[var], point, method = self.interp_method)[0]
        else:
            print(f'{var} is not a primitive, auxiliary variable or structure of the micro_model!!')

    @multimethod
    def get_var_gridpoint(self, var: str, h: object, i: object, j: object):
        """
        Returns quantity corresponding to input str 'var' evaluated 
        at the grid-point identified by grid indices h,i,j

        Parameters: 
        -----------
        var: str
            String corresponding to primitive, auxiliary or structure variable 

        h,i,j: int
            integers corresponding to position on the grid. 

        Returns: 
        --------
        Values or arrays corresponding to variable evaluated at grid-point corresponding to input 'h,i,j'. 

        Notes:
        ------
        This method is useful e.g. for plotting the raw data. 
        """
        if var in self.get_prim_strs():
            return self.prim_vars[var][h,i,j]
        elif var in self.get_aux_strs():
            return self.aux_vars[var][h,i,j]
        elif var in self.get_structures_strs():
            return self.structures[var][h,i,j]
        else: 
            print('{} is not a variable of model {}'.format(var, self.get_model_name()))
            return None

    @multimethod
    def get_var_gridpoint(self, var: str, point: object):
        """
        Returns quantity corresponding to input str 'var' evaluated 
        at at gridpoint closest to input 'point'.

        Parameters:
        -----------
        vars: string corresponding to primitive, auxiliary or structure variable

        point: list of 2+1 floats

        Returns: 
        --------
        Values or arrays corresponding to variable evaluated at the closest grid-point to input 'point'. 

        Notes:
        ------
        This method should be used in case using interpolated values 
        becomes too expensive. 
        """
        indices = Base.find_nearest_cell(point, self.domain_vars['points'])
        if var in self.get_prim_strs():
            return self.prim_vars[var][tuple(indices)]  
        elif var in self.get_aux_strs():
            return self.aux_vars[var][tuple(indices)]    
        elif var in self.get_structures_strs():
            return self.structures[var][tuple(indices)]
        else: 
            print(f"{var} is not a variable of the model!")
            return None

    def setup_structures(self):
        """
        Key routine: set up the structures (i.e baryon (mass) current BC, Stress-Energy tensors

        Notes:
        ------
        Structures are built as multi-dim np.arrays, with the first (three) indices referring 
        to the grid, while the last one or two refer to space-time components.
        
        The baryon current is stored as a fully contra-variant tensor (index up)
        The stress-energy tensor is stored as a fully contra-variant tensor (both indices up) 
        """
        self.structures["BC"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],3))
        self.structures["bar_vel"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],3))
        self.structures["SET"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],3,3))

        for h in range(self.domain_vars['nt']):
            for i in range(self.domain_vars['nx']):
                for j in range(self.domain_vars['ny']): 
                    vel_vec = np.array([self.aux_vars['W'][h,i,j],self.aux_vars['W'][h,i,j] * self.prim_vars['vx'][h,i,j] ,\
                                    self.aux_vars['W'][h,i,j] * self.prim_vars['vy'][h,i,j]])
                    
                    self.structures['bar_vel'][h,i,j,:] = vel_vec

                    self.structures['BC'][h,i,j,:] = np.multiply(self.prim_vars['n'][h,i,j], vel_vec )
                    
                    self.structures['SET'][h,i,j,:,:] = (self.prim_vars['n'][h,i,j] * self.aux_vars['h'][h,i,j]) * np.outer(vel_vec, vel_vec) \
                                                        + self.prim_vars['p'][h,i,j] * self.metric
                    
                    
                    

        self.vars = self.prim_vars
        self.vars.update(self.aux_vars)
        self.vars.update(self.structures)

class IdealHD_3D(object):
    """
    Class for storing data of 3D hydrodynamic simulations (of turbulence).  
    """

    def __init__(self, interp_method = "linear"):
        """
        Sets up the main variables and dictionaries. Var and dict names need not match
        those of the simulation output data, this can be taken into account using appropriate dictionaries of
        classes in FileReaders.py 

        Parameters
        ----------
        interp_method: str
            optional method to be used by interpn        
        """
        self.spatial_dims = 3
        self.interp_method = interp_method

        self.metric = np.zeros((4,4))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = self.metric[3,3] = +1

        #Dictionary for grid: info and points
        self.domain_int_strs = ('nt','nx','ny', 'nz')
        self.domain_float_strs = ("tmin","tmax","xmin","xmax","ymin","ymax","zmin","zmax","dt","dx","dy", "dz")
        self.domain_array_strs = ("t","x","y","z","points")
        self.domain_vars = dict.fromkeys(self.domain_int_strs+self.domain_float_strs+self.domain_array_strs)
        for str in self.domain_vars:
            self.domain_vars[str] = []   

        #Dictionary for primitive var
        self.prim_strs = ("vx","vy","vz","n","p")
        self.prim_vars = dict.fromkeys(self.prim_strs)
        for str in self.prim_strs:
            self.prim_vars[str] = []

        #Dictionary for auxiliary var
        self.aux_strs = ("W", "h", "e")
        self.aux_vars = dict.fromkeys(self.aux_strs)
        for str in self.aux_strs:
            self.aux_vars[str] = []

        #Dictionary for structures
        self.structures_strs = ("BC", "SET", "bar_vel")
        self.structures = dict.fromkeys(self.structures_strs)
        for str in self.structures_strs:
            self.structures[str] = []

        self.labels_var_dict = {'BC' : r'$n^{a}$', 
                                'SET' : r'$T^{ab}$', 
                                'bar_vel' : r'$u^a$',
                                'vx' : r'$v_x$', 
                                'vy' : r'$v_y$', 
                                'vz' : r'$v_z$', 
                                'n' : r'$n$', 
                                'W' : r'$W$', 
                                'e' : r'$e$', 
                                'h' : r'$h$', 
                                'p' : r'$p$'} 
        
    def get_spatial_dims(self):
        return self.spatial_dims

    def get_model_name(self):
        return 'Ideal Hydro (3+1d)'

    def get_domain_strs(self):
        return self.domain_int_strs + self.domain_float_strs + self.domain_array_strs
    
    def get_prim_strs(self):
        return self.prim_strs
    
    def get_aux_strs(self):
        return self.aux_strs
    
    def get_structures_strs(self):
        return self.structures_strs
    
    def get_all_var_strs(self):
        return self.get_prim_strs() + self.get_aux_strs() + self.get_structures_strs()
    
    def get_gridpoints(self): 
        return self.domain_vars['points']
    
    def get_interpol_var(self, var, point):
        """
        Returns quantity corresponding to input str 'var' evaluated 
        at the input point 'point' via interpolation.

        Parameters
        ----------
        var: str corresponding to primitive, auxiliary or structre variable
            
        point : list of floats
            ordered coordinates: t,x,y,z

        Return
        ------
        Interpolated values/arrays corresponding to variable. 
        Empty list if none of the variables is a primitive, auxiliary o structure of the micro_model

        Notes
        -----
        Interpolation may fail when too close to boundary
        """

        if var in self.get_prim_strs():
            return interpn(self.domain_vars['points'], self.prim_vars[var], point, method = self.interp_method)[0]
        elif var in self.get_aux_strs():
            return interpn(self.domain_vars['points'], self.aux_vars[var], point, method = self.interp_method)[0]
        elif var in self.get_structures_strs():
            return interpn(self.domain_vars['points'], self.structures[var], point, method = self.interp_method)[0]
        else:
            print(f'{var} is not a primitive, auxiliary variable or structure of the micro_model!!')

    @multimethod
    def get_var_gridpoint(self, var: str, h: object, i: object, j: object, k: object):
        """
        Returns quantity corresponding to input str 'var' evaluated 
        at the grid-point identified by grid indices h,i,j,k

        Parameters: 
        -----------
        var: str
            String corresponding to primitive, auxiliary or structure variable 

        h,i,j,k: int
            integers corresponding to position on the grid. 

        Returns: 
        --------
        Values or arrays corresponding to variable evaluated at grid-point corresponding to input 'h,i,j,k'. 

        Notes:
        ------
        This method is useful e.g. for plotting the raw data. 
        """
        if var in self.get_prim_strs():
            return self.prim_vars[var][h,i,j,k]
        elif var in self.get_aux_strs():
            return self.aux_vars[var][h,i,j,k]
        elif var in self.get_structures_strs():
            return self.structures[var][h,i,j,k]
        else: 
            print('{} is not a variable of model {}'.format(var, self.get_model_name()))
            return None

    @multimethod
    def get_var_gridpoint(self, var: str, point: object):
        """
        Returns quantity corresponding to input str 'var' evaluated 
        at at gridpoint closest to input 'point'.

        Parameters:
        -----------
        vars: string corresponding to primitive, auxiliary or structure variable

        point: list of 3+1 floats

        Returns: 
        --------
        Values or arrays corresponding to variable evaluated at the closest grid-point to input 'point'. 

        Notes:
        ------
        This method should be used in case using interpolated values 
        becomes too expensive. 
        """
        indices = Base.find_nearest_cell(point, self.domain_vars['points'])
        if var in self.get_prim_strs():
            return self.prim_vars[var][tuple(indices)]  
        elif var in self.get_aux_strs():
            return self.aux_vars[var][tuple(indices)]    
        elif var in self.get_structures_strs():
            return self.structures[var][tuple(indices)]
        else: 
            print(f"{var} is not a variable of the model!")
            return None
        
    def setup_structures(self):
        """
        Key routine: set up the structures, such as baryon (mass) current BC, Stress-Energy tensors

        Notes:
        ------
        Structures are built as multi-dim np.arrays, with the first (three) indices referring 
        to the grid, while the last one or two refer to space-time components.
        
        The baryon current is stored as a fully contra-variant tensor (index up)
        The stress-energy tensor is stored as a fully contra-variant tensor (both indices up) 
        """
        self.structures["BC"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'], 
                                          self.domain_vars['nz'], 4))
        self.structures["bar_vel"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],
                                               self.domain_vars['nz'], 4))
        self.structures["SET"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],
                                           self.domain_vars['nz'],4,4))

        for h in range(self.domain_vars['nt']):
            for i in range(self.domain_vars['nx']):
                for j in range(self.domain_vars['ny']): 
                    for k in range(self.domain_vars['nz']): 
                        vel_vec = np.array([self.aux_vars['W'][h,i,j,k],
                                        self.aux_vars['W'][h,i,j,k] * self.prim_vars['vx'][h,i,j,k] ,\
                                        self.aux_vars['W'][h,i,j,k] * self.prim_vars['vy'][h,i,j,k], 
                                        self.aux_vars['W'][h,i,j,k] * self.prim_vars['vz'][h,i,j,k]])
                        
                        self.structures['bar_vel'][h,i,j,k,:] = vel_vec

                        self.structures['BC'][h,i,j,k,:] = np.multiply(self.prim_vars['n'][h,i,j,k], vel_vec )
                        
                        self.structures['SET'][h,i,j,k,:,:] = (self.prim_vars['n'][h,i,j,k] * self.aux_vars['h'][h,i,j,k]) * np.outer(vel_vec, vel_vec) \
                                                            + self.prim_vars['p'][h,i,j,k] * self.metric
                        
                    
                    

        self.vars = self.prim_vars
        self.vars.update(self.aux_vars)
        self.vars.update(self.structures)

    ########################################################################
    # UNSUCCESFUL ATTEMPT TO PARALLELIZE SET-UP STRUCTURES: no speed-up 
    ########################################################################
    # @staticmethod
    # def setup_structures_task(W, vx, vy, vz, n, h, p, idxs):
    #     """
    #     Task to be executed in parallel to set-up the micro data

    #     Notes
    #     -----
    #     This becomes useful in the 3D case for sufficiently large datasets. 

    #     Test whether you get a speed up with an initializer (that sets the metric)
    #     """
    #     metric = np.zeros((4,4))
    #     metric[0,0]= -1
    #     metric[1,1] = metric[2,2] = metric[3,3] = +1
        
    #     bar_vel = np.array([W, W*vx, W*vy, W*vz])
    #     BC = np.multiply(n, bar_vel)
    #     SET = np.multiply(n * h, np.outer(bar_vel, bar_vel)) + np.multiply(p, metric)

    #     return bar_vel, BC, SET, idxs
    
    # def setup_structures_parallel(self, n_cpus):
        # """
        # werk
        # """
        # self.structures["BC"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'], 
        #                                   self.domain_vars['nz'], 4))
        # self.structures["bar_vel"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],
        #                                        self.domain_vars['nz'], 4))
        # self.structures["SET"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],
        #                                    self.domain_vars['nz'],4,4))
    
        # args_for_pool = []
        # for h in range(self.domain_vars['nt']):
        #     for i in range(self.domain_vars['nx']):
        #         for j in range(self.domain_vars['ny']):
        #             for k in range(self.domain_vars['nz']):
        #                     W = self.aux_vars['W'][h,i,j,k]
        #                     vx = self.prim_vars['vx'][h,i,j,k]
        #                     vy = self.prim_vars['vy'][h,i,j,k]
        #                     vz = self.prim_vars['vz'][h,i,j,k]
        #                     n = self.prim_vars['n'][h,i,j,k]
        #                     entalpy = self.aux_vars['h'][h,i,j,k]
        #                     p = self.prim_vars['p'][h,i,j,k]
        #                     args_for_pool.append((W, vx, vy, vz, n, entalpy, p, [h,i,j,k]))
                        
        # with mp.Pool(processes=n_cpus) as pool:
        #     print('Setting up micro structures in parallel with {} processes'.format(pool._processes), flush=True)
        #     for result in pool.starmap(IdealHD_3D.setup_structures_task, args_for_pool):
        #         h,i,j,k = tuple(result[3])
        #         self.structures['bar_vel'][h,i,j,k] = result[0]
        #         self.structures['BC'][h,i,j,k] = result[1]
        #         self.structures['SET'][h,i,j,k] = result[2]
                
class IdealMHD_3D(object):
    """
    Class for storing data of 3D magneto-hydrodynamic simulations (of turbulence). 
    """
    def __init__(self, interp_method = "linear"):
        """
        Sets up the main variables and dictionaries. Var and dict names need not match
        those of the simulation output data, this can be taken into account using appropriate dictionaries of
        classes in FileReaders.py 

        Parameters
        ----------
        interp_method: str
            optional method to be used by interpn        
        """
        self.spatial_dims = 3
        self.interp_method = interp_method

        self.metric = np.zeros((4,4))
        self.metric[0,0] = -1
        self.metric[1,1] = self.metric[2,2] = self.metric[3,3] = +1

        #Dictionary for grid: info and points
        self.domain_int_strs = ('nt','nx','ny', 'nz')
        self.domain_float_strs = ("tmin","tmax","xmin","xmax","ymin","ymax","zmin","zmax","dt","dx","dy", "dz")
        self.domain_array_strs = ("t","x","y","z","points")
        self.domain_vars = dict.fromkeys(self.domain_int_strs+self.domain_float_strs+self.domain_array_strs)
        for str in self.domain_vars:
            self.domain_vars[str] = []   

        #Dictionary for primitive var
        self.prim_strs = ("vx","vy","vz","n","p","Bx", "By", "Bz") #Bx etc is measured in the `lab-frame'
        self.prim_vars = dict.fromkeys(self.prim_strs)
        for str in self.prim_strs:
            self.prim_vars[str] = []

        #Dictionary for auxiliary var
        self.aux_strs = ("W", "h")
        self.aux_vars = dict.fromkeys(self.aux_strs)
        for str in self.aux_strs:
            self.aux_vars[str] = []

        #Dictionary for structures
        # The charge current is not in this list because the mesomodels we have in mind are in the 
        # framework of iMHD. As such storing the micro charge-current is not necessary. 
        self.structures_strs = ("BC", "SET", "Fab") #Fab is the Faraday tensor (not Maxwell)
        self.structures = dict.fromkeys(self.structures_strs)
        for str in self.structures_strs:
            self.structures[str] = []

        self.labels_var_dict = {'BC' : r'$n^{a}$', 
                                'SET' : r'$T^{ab}$', 
                                'Fab' : r'$F^{ab}$',
                                'vx' : r'$v_x$', 
                                'vy' : r'$v_y$', 
                                'vz' : r'$v_z$', 
                                'n' : r'$n$', 
                                'p' : r'$p$',
                                'Bx' : r'$B^x$',
                                'By' : r'$B^y$',
                                'Bz' : r'$B^z$',
                                'W' : r'$W$', 
                                'h' : r'$h$', 
                                }     

    def get_spatial_dims(self):
        return self.spatial_dims

    def get_model_name(self):
        return 'Ideal MHD (3+1d)'

    def get_domain_strs(self):
        return self.domain_int_strs + self.domain_float_strs + self.domain_array_strs
    
    def get_prim_strs(self):
        return self.prim_strs
    
    def get_aux_strs(self):
        return self.aux_strs
    
    def get_structures_strs(self):
        return self.structures_strs
    
    def get_all_var_strs(self):
        return self.get_prim_strs() + self.get_aux_strs() + self.get_structures_strs()
    
    def get_gridpoints(self): 
        return self.domain_vars['points']

    def get_interpol_var(self, var, point):
        """
        Returns quantity corresponding to input str 'var' evaluated 
        at the input point 'point' via interpolation.

        Parameters
        ----------
        var: str corresponding to primitive, auxiliary or structre variable
            
        point : list of floats
            ordered coordinates: t,x,y,z

        Return
        ------
        Interpolated values/arrays corresponding to variable. 
        Empty list if none of the variables is a primitive, auxiliary o structure of the micro_model

        Notes
        -----
        Interpolation may fail when too close to boundary
        """

        if var in self.get_prim_strs():
            return interpn(self.domain_vars['points'], self.prim_vars[var], point, method = self.interp_method)[0]
        elif var in self.get_aux_strs():
            return interpn(self.domain_vars['points'], self.aux_vars[var], point, method = self.interp_method)[0]
        elif var in self.get_structures_strs():
            return interpn(self.domain_vars['points'], self.structures[var], point, method = self.interp_method)[0]
        else:
            print(f'{var} is not a primitive, auxiliary variable or structure of the micro_model!!')

    @multimethod
    def get_var_gridpoint(self, var: str, h: object, i: object, j: object, k: object):
        """
        Returns quantity corresponding to input str 'var' evaluated 
        at the grid-point identified by grid indices h,i,j,k

        Parameters: 
        -----------
        var: str
            String corresponding to primitive, auxiliary or structure variable 

        h,i,j,k: int
            integers corresponding to position on the grid. 

        Returns: 
        --------
        Values or arrays corresponding to variable evaluated at grid-point corresponding to input 'h,i,j,k'. 

        Notes:
        ------
        This method is useful e.g. for plotting the raw data. 
        """
        if var in self.get_prim_strs():
            return self.prim_vars[var][h,i,j,k]
        elif var in self.get_aux_strs():
            return self.aux_vars[var][h,i,j,k]
        elif var in self.get_structures_strs():
            return self.structures[var][h,i,j,k]
        else: 
            print('{} is not a variable of model {}'.format(var, self.get_model_name()))
            return None

    @multimethod
    def get_var_gridpoint(self, var: str, point: object):
        """
        Returns quantity corresponding to input str 'var' evaluated 
        at at gridpoint closest to input 'point'.

        Parameters:
        -----------
        vars: string corresponding to primitive, auxiliary or structure variable

        point: list of 3+1 floats

        Returns: 
        --------
        Values or arrays corresponding to variable evaluated at the closest grid-point to input 'point'. 

        Notes:
        ------
        This method should be used in case using interpolated values 
        becomes too expensive. 
        """
        indices = Base.find_nearest_cell(point, self.domain_vars['points'])
        if var in self.get_prim_strs():
            return self.prim_vars[var][tuple(indices)]  
        elif var in self.get_aux_strs():
            return self.aux_vars[var][tuple(indices)]    
        elif var in self.get_structures_strs():
            return self.structures[var][tuple(indices)]
        else: 
            print(f"{var} is not a variable of the model!")
            return None
        
    def compute_current_point(self, h, i, j, k):
        """
        Left for future: to be implemented if meso models is resistive.

        Idea: Add separate routine to compute derivative of B. Then use here to compute 
        the curl of B. Set the spatial charge in the foliation equal to the curl of B. 
        Compute the charge density from the assumption of local charge neutrality as v*J 
        (in the foliation). Then return the four vector
        """
        pass

    def setup_structures(self):
        """
        Key routine: set up the structures, such as baryon (mass) current BC, Stress-Energy tensor
        and Faraday tensor. 

        Notes:
        ------
        Structures are built as multi-dim np.arrays, with the first (three) indices referring 
        to the grid, while the last one or two refer to space-time components.
        
        The baryon current is stored as a fully contra-variant tensor (index up)
        The stress-energy tensor is stored as a fully contra-variant tensor (both indices up) 
        The Faraday tensor is stored as a fully contravariant tensor (both indices up)
        """
        self.structures["BC"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'], 
                                          self.domain_vars['nz'],4))
        self.structures["SET"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],
                                           self.domain_vars['nz'],4,4))
        self.structures["Fab"] = np.zeros((self.domain_vars['nt'],self.domain_vars['nx'],self.domain_vars['ny'],
                                               self.domain_vars['nz'],4,4))

        for h in range(self.domain_vars['nt']):
            for i in range(self.domain_vars['nx']):
                for j in range(self.domain_vars['ny']): 
                    for k in range(self.domain_vars['nz']): 
                        vel_vec = np.array([self.aux_vars['W'][h,i,j,k],
                                        self.aux_vars['W'][h,i,j,k] * self.prim_vars['vx'][h,i,j,k] ,\
                                        self.aux_vars['W'][h,i,j,k] * self.prim_vars['vy'][h,i,j,k], 
                                        self.aux_vars['W'][h,i,j,k] * self.prim_vars['vz'][h,i,j,k]])

                        self.structures['BC'][h,i,j,k,:] = np.multiply(self.prim_vars['n'][h,i,j,k], vel_vec )


                        Bs = np.array([self.prim_vars['Bx'][h,i,j,k], self.prim_vars['By'][h,i,j,k], self.prim_vars['Bz'][h,i,j,k]])

                        b0 = np.dot(vel_vec[1:], Bs)
                        bi = np.multiply(1/self.aux_vars['W'][h,i,j,k], Bs) + np.multiply(b0/self.aux_vars['W'][h,i,j,k], vel_vec[1:]) 
                        ba = np.array([b0] + list(bi))
                        bsq = Base.Mink_dot(ba, ba)
                        # bsq = np.multiply( 1/(self.aux_vars['W']**2) , np.dot(Bs, Bs)) + np.multiply( 1/self.aux_vars['W'], np.dot(Bs,vel_vec[1:])) 
                        
                        rhohstar = np.multiply( self.prim_vars['n'] , self.aux_strs['h'][h,i,j,k] ) + bsq
                        pstar = self.prim_vars['p'] + bsq/2 


                        self.structures['SET'][h,i,j,k,:,:] = np.multiply(rhohstar , np.outer(vel_vec, vel_vec))  + np.multiply(pstar, self.metric ) \
                                                                - np.outer(ba, ba)
                        
                        Maxwell_contrav = np.outer(ba, vel_vec) - np.outer(vel_vec, ba)
                        Maxwell_covar = np.einsum('ij,kl,jl->ik', self.metric, self.metric, Maxwell_contrav )

                        self.structures['Fab'][h,i,j,k,:,:] = np.multiply(1/2, np.einsum('ijkl,kl->ij', self.Levi4D, Maxwell_covar) )
                        

                        # Here is missing the charge four current. This would have to be stored and filtered if the 
                        # meso model considered is resistive. We for now consider mesomodels within the framework of iMHD
                        # and then do not compute nor store it. 
                        # self.structures['ja'][h,i,j,k,:] = self.compute_Current_point(h,i,j,k)
                    
                    

        self.vars = self.prim_vars
        self.vars.update(self.aux_vars)
        self.vars.update(self.structures)



if __name__ == '__main__':

    eos_para = {"polytrope_K": 4.897 * 1e14, "Gamma_b": 1.31, "Gamma_th": 1.5}

    Aenus_reader = Aenus3D_h5py()
    micromodel = IdealMHD_3D()

    directory = "/Volumes/Seagate/Work/Miquel's/" + "mri-1225.h5"
    enclosed_grid = False
    res = (50,50,50)

    # read_in_data = timer_decorator(Aenus_reader.read_in_data)
    # read_in_data(directory, enclosed_grid, res, eos_para, micromodel)
    read_in_data_parallel = timer_decorator(Aenus_reader.read_in_data_parallel)
    read_in_data_parallel(directory, enclosed_grid, res, eos_para, micromodel, 8)