# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:00:02 2023

@author: Thomas
"""

import numpy as np
import math
import time
import os
import multiprocessing as mp
from itertools import product

from scipy.interpolate import interpn 
from multimethod import multimethod

from FileReaders import *
from system.BaseFunctionality import *


# These are the symbols, so be careful when using these to construct vectors!
# levi3D = np.array([[[ np.sign(i-j) * np.sign(j- k) * np.sign(k-i) \
#                       for k in range(3)]for j in range(3) ] for i in range(3) ])

# levi4D = np.array([[[[ np.sign(i - j) * np.sign(j - k) * np.sign(k - l) * np.sign(i - l) \
#                        for l in range(4)] for k in range(4) ] for j in range(4)] for i in range(4)])

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
        Values or arrays corresponding to variable evaluated at the closest grid-point to input 'point'. 

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
            print(f"{var} is not a variable of IdealMHD_2D!")
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

