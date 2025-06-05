# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:00:00 2023

@authors: Thomas & Marcus 
"""

import h5py
import glob
import numpy as np
import math 
import multiprocessing as mp

from scipy.interpolate import interpn

class METHOD_HDF5(object):

    def __init__(self, directory, fewer_snaps=False, smaller_list=None):
        """
        Set up the list of files (from hdf5) and dictionary with dataset names
        in the hdf5 file. Use 'fewer_snaps' and 'smaller_list' if the directory 
        contains more snapshot than needed. 

        Parameters
        ----------
        directory: string 
            the filenames in the directory have to be incremental (sorted is used)

        fewer_snaps: bool
            set to true if you want to store fewer snapshots than found in directory 
        
        smaller_list: list
            indices to be retained of list orderd via sorted(glob.glob(directory+str('*.hdf5')))

        """ 
        hdf5_filenames = sorted(glob.glob(directory+str('*.hdf5')))
        if fewer_snaps:
            if smaller_list:
                temp = [hdf5_filenames[i] for i in smaller_list]
                hdf5_filenames = temp

        self.hdf5_files = []
        for filename in hdf5_filenames:
            self.hdf5_files.append(h5py.File(filename,'r'))
        self.num_files = len(self.hdf5_files)

        self.hdf5_keys = dict.fromkeys(list(self.hdf5_files[0].keys())) 
        for key in self.hdf5_keys: 
            self.hdf5_keys[key] = list(self.hdf5_files[0][key].keys())

    def get_hdf5_keys(self):
        """
        Return the keys of the dictionary with the stored data from HDF5
        """
        return self.hdf5_keys

    def read_in_data(self, micro_model):   
        """
        Store data from files into micro_model 

        Parameters
        ----------
        micro_model: instance of a MicroModel 
            strs in micromodel have to be the same as hdf5 files output from METHOD.

        Notes 
        -----
            use and adapt translating dictionary if you want to store data in micromodel 
            using different keys than those from METHOD.

        """ 

        self.translating_prims = dict.fromkeys(micro_model.get_prim_strs())
        for prim_str in micro_model.get_prim_strs():
            if prim_str == "n":
                self.translating_prims[prim_str] = "rho"
            else: 
                self.translating_prims[prim_str] = prim_str 

        for prim_var_str in  micro_model.prim_vars:
            try: 
                method_str = self.translating_prims[prim_var_str]
                for counter in range(self.num_files):
                    micro_model.prim_vars[prim_var_str].append( self.hdf5_files[counter]["Primitive/"+method_str][:] )
                    # The [:] is for returning the arrays not the dataset
                micro_model.prim_vars[prim_var_str]  = np.array(micro_model.prim_vars[prim_var_str])
            except KeyError:
                print(f'{method_str} is not in the hdf5 dataset: check Primitive/')
        

        self.translating_aux = dict.fromkeys(micro_model.get_aux_strs())
        for aux_str in micro_model.get_aux_strs():
            self.translating_aux[aux_str] = aux_str

        for aux_var_str in  micro_model.aux_vars:
            try: 
                method_str = self.translating_aux[aux_var_str]
                for counter in range(self.num_files):
                    micro_model.aux_vars[aux_var_str].append( self.hdf5_files[counter]["Auxiliary/"+method_str][:] )
                micro_model.aux_vars[aux_var_str] = np.array(micro_model.aux_vars[aux_var_str])
            except KeyError:
                print(f'{method_str} is not in the hdf5 dataset: check Auxiliary/')
 
        # As METHOD saves endTime, the time variables (and points) need to be dealt with separately
        for dom_var_str in micro_model.domain_int_strs: 
            try: 
                if dom_var_str == 'nt': 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = int( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        for dom_var_str in micro_model.domain_float_strs: 
            try: 
                if dom_var_str in ['tmin', 'tmax']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = float( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        for dom_var_str in micro_model.domain_array_strs: 
            try: 
                if dom_var_str in ['t','points']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = self.hdf5_files[0]['Domain/' + dom_var_str][:]
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')


        micro_model.domain_vars['nt'] = self.num_files
        for counter in range(self.num_files):
            micro_model.domain_vars['t'].append( float(self.hdf5_files[counter]['Domain/endTime'][:]))
        micro_model.domain_vars['t'] = np.array(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmin'] = np.amin(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmax'] = np.amax(micro_model.domain_vars['t'])
        micro_model.domain_vars['points'] = [micro_model.domain_vars['t'], micro_model.domain_vars['x'], \
                                             micro_model.domain_vars['y']]

    def read_in_data_HDF5_missing_xy(self, micro_model):   
        """
        Store data from files into micro_model. To be used if METHOD output does 
        not contain explicitly the grid points

        Parameters
        ----------
        micro_model: instance of a MicroModel 
            strs in micromodel have to be the same as hdf5 files output from METHOD.

        Notes 
        -----
            use and adapt translating dictionary if you want to store data in micromodel 
            using different keys than those from METHOD.
        """ 

        self.translating_prims = dict.fromkeys(micro_model.get_prim_strs())
        for prim_str in micro_model.get_prim_strs():
            if prim_str == "n":
                self.translating_prims[prim_str] = "rho"
            else: 
                self.translating_prims[prim_str] = prim_str 

        for prim_var_str in  micro_model.prim_vars:
            try: 
                method_str = self.translating_prims[prim_var_str]
                for counter in range(self.num_files):
                    micro_model.prim_vars[prim_var_str].append( self.hdf5_files[counter]["Primitive/"+method_str][:] )
                    # The [:] is for returning the arrays not the dataset
                micro_model.prim_vars[prim_var_str]  = np.array(micro_model.prim_vars[prim_var_str])
            except KeyError:
                print(f'{method_str} is not in the hdf5 dataset: check Primitive/')
        

        self.translating_aux = dict.fromkeys(micro_model.get_aux_strs())
        for aux_str in micro_model.get_aux_strs():
            self.translating_aux[aux_str] = aux_str

        for aux_var_str in  micro_model.aux_vars:
            try: 
                method_str = self.translating_aux[aux_var_str]
                for counter in range(self.num_files):
                    micro_model.aux_vars[aux_var_str].append( self.hdf5_files[counter]["Auxiliary/"+method_str][:] )
                micro_model.aux_vars[aux_var_str] = np.array(micro_model.aux_vars[aux_var_str])
            except KeyError:
                print(f'{method_str} is not in the hdf5 dataset: check Auxiliary/')
 
        # As METHOD saves endTime, the time variables (and points) need to be dealt with separately
        # Similar is for x,y which are not stored by METHOD parallelSaveDataHDF5
        for dom_var_str in micro_model.domain_int_strs: 
            try: 
                if dom_var_str == 'nt': 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = int( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        for dom_var_str in micro_model.domain_float_strs: 
            try: 
                if dom_var_str in ['tmin', 'tmax']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = float( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        for dom_var_str in micro_model.domain_array_strs: 
            try: 
                if dom_var_str in ['t','points', 'x', 'y']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = self.hdf5_files[0]['Domain/' + dom_var_str][:]
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')


        micro_model.domain_vars['nt'] = self.num_files
        for counter in range(self.num_files):
            micro_model.domain_vars['t'].append( float(self.hdf5_files[counter]['Domain/endTime'][:]))
        micro_model.domain_vars['t'] = np.array(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmin'] = np.amin(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmax'] = np.amax(micro_model.domain_vars['t'])

        micro_model.domain_vars['x'] = np.zeros(micro_model.domain_vars['nx'])
        for i in range(len(micro_model.domain_vars['x'])):
            micro_model.domain_vars['x'][i] = micro_model.domain_vars['xmin'] + i * micro_model.domain_vars['dx']
            
        micro_model.domain_vars['y'] = np.zeros(micro_model.domain_vars['ny'])
        for i in range(len(micro_model.domain_vars['y'])):
            micro_model.domain_vars['y'][i] = micro_model.domain_vars['ymin'] + i * micro_model.domain_vars['dy']

        micro_model.domain_vars['points'] = [micro_model.domain_vars['t'], micro_model.domain_vars['x'], \
                                             micro_model.domain_vars['y']]

    def read_in_data3D(self, micro_model):   
        """
        Store 3+1 dimensional data from files into micro_model

        Parameters
        ----------
        micro_model: instance of a MicroModel 
            strs in micromodel have to be the same as hdf5 files output from METHOD.

        Notes 
        -----
            use and adapt translating dictionary if you want to store data in micromodel 
            using different keys than those from METHOD.

        """ 

        self.translating_prims = dict.fromkeys(micro_model.get_prim_strs())
        for prim_str in micro_model.get_prim_strs():
            if prim_str == "n":
                self.translating_prims[prim_str] = "rho"
            else: 
                self.translating_prims[prim_str] = prim_str 

        for prim_var_str in  micro_model.prim_vars:
            try: 
                method_str = self.translating_prims[prim_var_str]
                for counter in range(self.num_files):
                    micro_model.prim_vars[prim_var_str].append( self.hdf5_files[counter]["Primitive/"+method_str][:] )

                micro_model.prim_vars[prim_var_str]  = np.array(micro_model.prim_vars[prim_var_str])
            except KeyError:
                print(f'{method_str} is not in the hdf5 dataset: check Primitive/')
        

        self.translating_aux = dict.fromkeys(micro_model.get_aux_strs())
        for aux_str in micro_model.get_aux_strs():
            self.translating_aux[aux_str] = aux_str

        for aux_var_str in  micro_model.aux_vars:
            try: 
                method_str = self.translating_aux[aux_var_str]
                for counter in range(self.num_files):
                    micro_model.aux_vars[aux_var_str].append( self.hdf5_files[counter]["Auxiliary/"+method_str][:] )
                micro_model.aux_vars[aux_var_str] = np.array(micro_model.aux_vars[aux_var_str])
            except KeyError:
                print(f'{method_str} is not in the hdf5 dataset: check Auxiliary/')
 
        # As METHOD saves endTime, the time variables (and points) need to be dealt with separately
        for dom_var_str in micro_model.domain_int_strs: 
            try: 
                if dom_var_str == 'nt': 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = int( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        for dom_var_str in micro_model.domain_float_strs: 
            try: 
                if dom_var_str in ['tmin', 'tmax']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = float( self.hdf5_files[0]['Domain/' + dom_var_str][:])
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')

        for dom_var_str in micro_model.domain_array_strs: 
            try: 
                if dom_var_str in ['t','points', 'x', 'y', 'z']: 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = self.hdf5_files[0]['Domain/' + dom_var_str][:]
            except KeyError: 
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')


        micro_model.domain_vars['nt'] = self.num_files
        for counter in range(self.num_files):
            micro_model.domain_vars['t'].append( float(self.hdf5_files[counter]['Domain/endTime'][:]))

        micro_model.domain_vars['x'] = np.zeros(micro_model.domain_vars['nx'])
        for i in range(len(micro_model.domain_vars['x'])):
            micro_model.domain_vars['x'][i] = micro_model.domain_vars['xmin'] + i * micro_model.domain_vars['dx']

        micro_model.domain_vars['y'] = np.zeros(micro_model.domain_vars['ny'])
        for i in range(len(micro_model.domain_vars['y'])):
            micro_model.domain_vars['y'][i] = micro_model.domain_vars['ymin'] + i * micro_model.domain_vars['dy']

        micro_model.domain_vars['z'] = np.zeros(micro_model.domain_vars['nz'])
        for i in range(len(micro_model.domain_vars['z'])):
            micro_model.domain_vars['z'][i] = micro_model.domain_vars['zmin'] + i * micro_model.domain_vars['dz']

            
        micro_model.domain_vars['t'] = np.array(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmin'] = np.amin(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmax'] = np.amax(micro_model.domain_vars['t'])
        micro_model.domain_vars['points'] = [micro_model.domain_vars['t'], micro_model.domain_vars['x'], \
                                             micro_model.domain_vars['y'], micro_model.domain_vars['z']]
        

class Aenus3D_h5py(object):
    """
    Werk: need to think about the case with many snapshots...
    """
    def __init__(self):
        """
        Nothing really...
        """
        self.spatial_dims = 3

        # factors to convert from cgs to geometric units
        self.length_factor = 1 / (1.47651 * 1e5) 
        self.time_factor = 1/ (4.92513 * 1e-6) 
        self.B_factor = 1 / 1e20
        self.velocity_factor = 1 / (2.9979 * 1e10)
        self.density_factor = 1 / 6.17714 * 1e17
        self.pressure_factor = 1 / (5.55173 * 1e38)
        
        self.light_speed = 2.9979 * 1e10

        # PASS THIS AS DICTIONARY WHEN READING THE DATA
        # self.polytrope_K = 4.897 * 1e14  #in cgs
        # self.Gamma_b = 1.31
        # self.Gamma_th = 1.5

        
    def cyl2cart_small(self, Rs, phis, zs):
        """
        Compute cartesian grid ranges such that it is fully enclosed by the the cylindrical-grid

        Parameters
        ----------
        Rs, phis, zs: np.array:
        the coordinates of the cylindrical grid

        Returns
        -------
        Dictionary with keys: x_range, y_range, z_range
        """
        assert np.mean(phis) < 1e-11

        x_m = np.amax(np.amin(Rs) * np.cos(phis))
        x_M = np.amin(np.amax(Rs) * np.cos(phis))
        y_m = np.amin(np.amin(Rs) * np.sin(phis))
        y_M = np.amax(np.amin(Rs) * np.sin(phis))
        z_m = np.amin(zs)
        z_M = np.amax(zs)

        Xs = np.einsum('i,j->ij',Rs,np.cos(phis))
        Ys = np.einsum('i,j->ij',Rs,np.sin(phis))

        assert x_m >= np.amin(Xs)
        assert x_M <= np.amax(Xs)
        assert y_m >= np.amin(Ys)
        assert y_M <= np.amax(Ys)
        
        return {'x_range': [x_m, x_M], 'y_range': [y_m, y_M], 'z_range': [z_m, z_M]}

    def cyl2cart_large(self, Rs, phis, zs):
        """
        Compute Cartesian grid ranges such that the cylindrical-grid is fully enclosed by the cartesian grid

        Parameters
        ----------
        Rs, phis, zs: np.array:
        the coordinates of the cylindrical grid

        Returns
        -------
        Dictionary with keys: x_range, y_range, z_range
        """
        Xs = np.einsum('i,j->ij',Rs,np.cos(phis))
        Ys = np.einsum('i,j->ij',Rs,np.sin(phis))

        x_m = np.amin(Xs)
        x_M = np.amax(Xs)
        y_m = np.amin(Ys)
        y_M = np.amax(Ys)

        z_m = np.amin(zs)
        z_M = np.amax(zs)

        return {'x_range': [x_m, x_M], 'y_range': [y_m, y_M], 'z_range': [z_m, z_M]}
    
    def read_in_data(self, files_dir, enclosed_grid, res, eos_para, micro_model, method='linear'):
        """
        Serial routine to read in data from Aenus sim, set up and fill in the micromodel Cartesian grid via interpolation. 

        Parameters
        ----------
        files_dir: string 
        path to file with data
        
        enclosed_grid: bool
        If True the cartesian grid will be fully contained in the cylindrical one. 

        res: tuple(int, int, int)
        the resolution of the Cartesian grid

        eos_para: dict with keys ["polytrope_K", "Gamma_b", "Gamma_th"]
        
        micro_model: instance of a micromodel class where to store data
         
        method: string
        interpolation method to be used to fill in the Cartesian grid.
        """
        # compatibility check: the eos_para dictionary
        same_keys = set(eos_para.keys()) == set(["polytrope_K", "Gamma_b", "Gamma_th"])
        if not same_keys:
            print("Problem with the dictionary of EoS parameters, exiting.")
            return None


        # Reading the simulation output
        time, r, phi, z, br, bphi, bz, vr, vphi, vz, Pgas, rho = [], [], [], [], [], [], [], [], [], [], [], []
        # gravpot: do we need the gravitational potential? don't think so. 

        with h5py.File(files_dir, 'r') as file:
            time = np.array(file['time'][:]) 
            r = np.array(file['radius'][:])
            phi = np.array(file['phi'][:])
            z = np.array(file['z'][:])
            br = np.array(file['br'][:])
            bphi = np.array(file['bphi'][:])
            bz = np.array(file['bz'][:])
            vr = np.array(file['vr'][:])
            vphi = np.array(file['vphi'][:])
            vz = np.array(file['vz'][:])
            # gravpot: do we need the gravitational potential? don't think so. 
            Pgas = np.array(file['Pgas'][:])
            rho = np.array(file['rho'][:])

        # rescaling to geometric units: Aenus output is in cgs
        time *= self.time_factor
        r *= self.length_factor
        z *= self.length_factor
        br *= self.B_factor
        bphi *= self.B_factor
        bz *= self.B_factor
        vr *= self.velocity_factor
        vphi *= self.velocity_factor
        vz *= self.velocity_factor
        # gravpot: do we need the gravitational potential? don't think so. 
        Pgas *= self.pressure_factor
        rho *= self.density_factor

        # Creating the cartesian grid and setting up domain vars
        if enclosed_grid:
            grid_ranges = self.cyl2cart_small(r, phi, z)
            bounds_error = True
            fill_value = None
        else: 
            grid_ranges = self.cyl2cart_large(r, phi, z)
            bounds_error = False
            fill_value = None

        # s = (enclosed_grid==True) ? 'contained in' : 'containing'
        s = "enclosed by" if enclosed_grid==True else "containing"
        print(f'Interpolating on the cartesian grid {s} the cylindrical one.')
        print(f'Cylindrical grid size: {len(r)}, {len(phi)}, {len(z)} (R,phi,z).')
        print(f'Cartesian grid size: {res[0]}, {res[1]}, {res[2]} (x,y,z).')

        print(grid_ranges)


        micro_model.domain_vars['t'] = time
        micro_model.domain_vars['x'] = np.linspace(grid_ranges['x_range'][0], grid_ranges['x_range'][1], res[0])
        micro_model.domain_vars['y'] = np.linspace(grid_ranges['y_range'][0], grid_ranges['y_range'][1], res[1])
        micro_model.domain_vars['z'] = np.linspace(grid_ranges['z_range'][0], grid_ranges['z_range'][1], res[2])
        micro_model.domain_vars['points'] = [micro_model.domain_vars['t'], micro_model.domain_vars['x'], micro_model.domain_vars['y'], micro_model.domain_vars['z']]

        micro_model.domain_vars['nt'] = len(micro_model.domain_vars['t'])
        micro_model.domain_vars['nx'] = len(micro_model.domain_vars['x'])
        micro_model.domain_vars['ny'] = len(micro_model.domain_vars['y'])
        micro_model.domain_vars['nz'] = len(micro_model.domain_vars['z'])

        micro_model.domain_vars['tmin'] = np.amin(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmax'] = np.amax(micro_model.domain_vars['t'])
        micro_model.domain_vars['xmin'] = np.amin(micro_model.domain_vars['x'])
        micro_model.domain_vars['xmax'] = np.amax(micro_model.domain_vars['x'])
        micro_model.domain_vars['ymin'] = np.amin(micro_model.domain_vars['y'])
        micro_model.domain_vars['ymax'] = np.amax(micro_model.domain_vars['y'])
        micro_model.domain_vars['zmin'] = np.amin(micro_model.domain_vars['z'])
        micro_model.domain_vars['zmax'] = np.amax(micro_model.domain_vars['z'])

        micro_model.domain_vars['dt'] = (micro_model.domain_vars['tmax'] - micro_model.domain_vars['tmin']) / micro_model.domain_vars['nt']
        micro_model.domain_vars['dx'] = (micro_model.domain_vars['xmax'] - micro_model.domain_vars['xmin']) / micro_model.domain_vars['nx']
        micro_model.domain_vars['dy'] = (micro_model.domain_vars['ymax'] - micro_model.domain_vars['ymin']) / micro_model.domain_vars['ny']
        micro_model.domain_vars['dz'] = (micro_model.domain_vars['zmax'] - micro_model.domain_vars['zmin']) / micro_model.domain_vars['nz']

        # Setting up the fields
        shape = (micro_model.domain_vars['nt'], micro_model.domain_vars['nx'], micro_model.domain_vars['ny'], micro_model.domain_vars['nz'])

        micro_model.prim_vars['Bx'] = np.zeros(shape)
        micro_model.prim_vars['By'] = np.zeros(shape)
        micro_model.prim_vars['Bz'] = np.zeros(shape)
        micro_model.prim_vars['vx'] = np.zeros(shape)
        micro_model.prim_vars['vy'] = np.zeros(shape)
        micro_model.prim_vars['vz'] = np.zeros(shape)
        micro_model.prim_vars['n'] = np.zeros(shape)
        micro_model.prim_vars['p'] = np.zeros(shape)
    
        # Interpolating from cylindrical grid
        for h, T in enumerate(micro_model.domain_vars['t']):
            for i, X in enumerate(micro_model.domain_vars['x']):
                for j, Y in enumerate(micro_model.domain_vars['y']):
                    for k, Z in enumerate(micro_model.domain_vars['z']):
                        R = np.sqrt(X**2 + Y**2)
                        PHI = math.atan2(Y,X)

                        micro_model.prim_vars['n'][h,i,j,k] = interpn([r, phi, z], rho, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]
                        micro_model.prim_vars['p'][h,i,j,k] = interpn([r, phi, z], Pgas, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]

                        
                        BR = interpn([r, phi, z], br, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]
                        Bphi = interpn([r, phi, z], bphi, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]
                        Bx = BR * np.cos(PHI) + Bphi * np.sin(PHI)
                        By = BR * np.sin(PHI) + Bphi * np.cos(PHI)
                        micro_model.prim_vars['Bx'][h,i,j,k] = Bx
                        micro_model.prim_vars['By'][h,i,j,k] = By
                        micro_model.prim_vars['Bz'][h,i,j,k] = interpn([r, phi, z], bz, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]

                        
                        VR = interpn([r, phi, z], vr, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]
                        Vphi = interpn([r, phi, z], vphi, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]
                        Vx = VR * np.cos(PHI) + Vphi * np.sin(PHI)
                        Vy = VR * np.sin(PHI) + Vphi * np.cos(PHI)
                        micro_model.prim_vars['vx'][h,i,j,k] = Vx
                        micro_model.prim_vars['vy'][h,i,j,k] = Vy
                        micro_model.prim_vars['vz'][h,i,j,k] = interpn([r, phi, z], vz, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]

        # Setting up the auxiliary variables: Lorentz factor 
        W = (micro_model.prim_vars['vx']**2 + micro_model.prim_vars['vy']**2 + micro_model.prim_vars['vz']**2)
        W = 1. / np.sqrt(1-W)
        micro_model.aux_vars['W'] = W

        # Setting up the enthalpy: need to take care of dimensionality of the polytropic constant. 
        polytrope_K = eos_para['polytrope_K']
        Gamma_b = eos_para['Gamma_b']
        Gamma_th = eos_para['Gamma_th']
        Pth = (micro_model.prim_vars['p'] / self.pressure_factor) - polytrope_K * np.power(micro_model.prim_vars['n'] * self.density_factor, Gamma_b) #thermal pressure in cgs
        Pth = Pth * self.pressure_factor # thermal pressure in geometrized units
        Pb = micro_model.prim_vars['p'] - Pth
        h = micro_model.prim_vars['p'] / micro_model.prim_vars['n'] + Pth / (Gamma_th-1) + Pb/(Gamma_b-1)
        micro_model.aux_vars['h'] = h  

        micro_model.aux_vars.update(eos_para)

    @staticmethod
    def RID_initializer(eenclosed_grid, ccyl_grid, rrho, PPgas, bbr, bbphi, bbz, vvr, vvphi, vvz, mmethod):
        """
        Initializer for read_in_data_parallel
        """

        global enclosed_grid
        global cyl_grid 
        global rho 
        global Pgas 
        global br
        global bphi
        global bz 
        global vr 
        global vphi
        global vz 
        global method

        enclosed_grid = eenclosed_grid
        method = mmethod
        cyl_grid = ccyl_grid
        rho = rrho
        Pgas = PPgas
        br = bbr
        bphi = bbphi
        bz = bbz
        vr = vvr
        vphi = vvphi
        vz = vvz

    @staticmethod
    def task_RID(point, idxs):
        """
        Task to be executed in parallel by read_in_data_parallel

        Parameters
        ----------

        point = [t, x, y, z]

        idxs = [h, i, j, k]

        Returns
        -------
        idxs, n, P, Bx, By, Bz, Vx, Vy, Vz
        """
        global enclosed_grid
        global method
        global cyl_grid 
        global rho 
        global Pgas 
        global br
        global bphi
        global bz 
        global vr 
        global vphi
        global vz 

        if enclosed_grid:
            bounds_error = True
            fill_value = None
        else: 
            bounds_error = False
            fill_value = None

        t,x,y,z = point
        R = np.sqrt(x**2 + y**2)
        PHI = math.atan2(y,x)
        Z = z
        
        n = interpn(cyl_grid, rho, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]
        P = interpn(cyl_grid, Pgas, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]

        BR = interpn(cyl_grid, br, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]
        Bphi = interpn(cyl_grid, bphi, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]
        Bx = BR * np.cos(PHI) + Bphi * np.sin(PHI)
        By = BR * np.sin(PHI) + Bphi * np.cos(PHI)
        Bz = interpn(cyl_grid, bz, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]

        VR = interpn(cyl_grid, br, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]
        Vphi = interpn(cyl_grid, bphi, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]
        Vx = VR * np.cos(PHI) + Vphi * np.sin(PHI)
        Vy = VR * np.sin(PHI) + Vphi * np.cos(PHI)
        Vz = interpn(cyl_grid, bz, [R,PHI,Z], method = method, bounds_error = bounds_error, fill_value = fill_value)[0]

        return idxs, n, P, Bx, By, Bz, Vx, Vy, Vz

    def read_in_data_parallel(self, files_dir, enclosed_grid, res, eos_para, micro_model, n_cpus, method='linear'):
        """
        Parallelized version of read_in_data()

        Parameters
        ----------
        files_dir: string 
        path to file with data
        
        enclosed_grid: bool
        If True the cartesian grid will be fully contained in the cylindrical one. 

        res: tuple(int, int, int)
        the resolution of the Cartesian grid

        eos_para: dict with keys ["polytrope_K", "Gamma_b", "Gamma_th"]
        
        micro_model: instance of a micromodel class where to store data

        n_cpus: integer
        number of processes for pool.starmap
         
        method: string
        interpolation method to be used to fill in the Cartesian grid.
        """
        # compatibility check: the eos_para dictionary
        same_keys = set(eos_para.keys()) == set(["polytrope_K", "Gamma_b", "Gamma_th"])
        if not same_keys:
            print("Problem with the dictionary of EoS parameters, exiting.")
            return None


        # Reading the simulation output
        time, r, phi, z, br, bphi, bz, vr, vphi, vz, Pgas, rho = [], [], [], [], [], [], [], [], [], [], [], []
        # gravpot: do we need the gravitational potential? don't think so. 

        with h5py.File(files_dir, 'r') as file:
            time = np.array(file['time'][:]) 
            r = np.array(file['radius'][:])
            phi = np.array(file['phi'][:])
            z = np.array(file['z'][:])
            br = np.array(file['br'][:])
            bphi = np.array(file['bphi'][:])
            bz = np.array(file['bz'][:])
            vr = np.array(file['vr'][:])
            vphi = np.array(file['vphi'][:])
            vz = np.array(file['vz'][:])
            # gravpot: do we need the gravitational potential? don't think so. 
            Pgas = np.array(file['Pgas'][:])
            rho = np.array(file['rho'][:])

        # rescaling to geometric units: Aenus output is in cgs
        time *= self.time_factor
        r *= self.length_factor
        z *= self.length_factor
        br *= self.B_factor
        bphi *= self.B_factor
        bz *= self.B_factor
        vr *= self.velocity_factor
        vphi *= self.velocity_factor
        vz *= self.velocity_factor
        # gravpot: do we need the gravitational potential? don't think so. 
        Pgas *= self.pressure_factor
        rho *= self.density_factor

        # Creating the cartesian grid and setting up domain vars
        if enclosed_grid:
            grid_ranges = self.cyl2cart_small(r, phi, z)
            bounds_error = True
            fill_value = None
        else: 
            grid_ranges = self.cyl2cart_large(r, phi, z)
            bounds_error = False
            fill_value = None

        # s = (enclosed_grid==True) ? 'contained in' : 'containing'
        s = "enclosed by" if enclosed_grid==True else "containing"
        print(f'Interpolating on the cartesian grid {s} the cylindrical one.')
        print(f'Cylindrical grid size: {len(r)}, {len(phi)}, {len(z)} (R,phi,z).')
        print(f'Cartesian grid size: {res[0]}, {res[1]}, {res[2]} (x,y,z).')

        print(grid_ranges)

        micro_model.domain_vars['t'] = time
        micro_model.domain_vars['x'] = np.linspace(grid_ranges['x_range'][0], grid_ranges['x_range'][1], res[0])
        micro_model.domain_vars['y'] = np.linspace(grid_ranges['y_range'][0], grid_ranges['y_range'][1], res[1])
        micro_model.domain_vars['z'] = np.linspace(grid_ranges['z_range'][0], grid_ranges['z_range'][1], res[2])
        micro_model.domain_vars['points'] = [micro_model.domain_vars['t'], micro_model.domain_vars['x'], micro_model.domain_vars['y'], micro_model.domain_vars['z']]

        micro_model.domain_vars['nt'] = len(micro_model.domain_vars['t'])
        micro_model.domain_vars['nx'] = len(micro_model.domain_vars['x'])
        micro_model.domain_vars['ny'] = len(micro_model.domain_vars['y'])
        micro_model.domain_vars['nz'] = len(micro_model.domain_vars['z'])

        micro_model.domain_vars['tmin'] = np.amin(micro_model.domain_vars['t'])
        micro_model.domain_vars['tmax'] = np.amax(micro_model.domain_vars['t'])
        micro_model.domain_vars['xmin'] = np.amin(micro_model.domain_vars['x'])
        micro_model.domain_vars['xmax'] = np.amax(micro_model.domain_vars['x'])
        micro_model.domain_vars['ymin'] = np.amin(micro_model.domain_vars['y'])
        micro_model.domain_vars['ymax'] = np.amax(micro_model.domain_vars['y'])
        micro_model.domain_vars['zmin'] = np.amin(micro_model.domain_vars['z'])
        micro_model.domain_vars['zmax'] = np.amax(micro_model.domain_vars['z'])

        micro_model.domain_vars['dt'] = (micro_model.domain_vars['tmax'] - micro_model.domain_vars['tmin']) / micro_model.domain_vars['nt']
        micro_model.domain_vars['dx'] = (micro_model.domain_vars['xmax'] - micro_model.domain_vars['xmin']) / micro_model.domain_vars['nx']
        micro_model.domain_vars['dy'] = (micro_model.domain_vars['ymax'] - micro_model.domain_vars['ymin']) / micro_model.domain_vars['ny']
        micro_model.domain_vars['dz'] = (micro_model.domain_vars['zmax'] - micro_model.domain_vars['zmin']) / micro_model.domain_vars['nz']

        # Setting up the fields
        shape = (micro_model.domain_vars['nt'], micro_model.domain_vars['nx'], micro_model.domain_vars['ny'], micro_model.domain_vars['nz'])

        micro_model.prim_vars['Bx'] = np.zeros(shape)
        micro_model.prim_vars['By'] = np.zeros(shape)
        micro_model.prim_vars['Bz'] = np.zeros(shape)
        micro_model.prim_vars['vx'] = np.zeros(shape)
        micro_model.prim_vars['vy'] = np.zeros(shape)
        micro_model.prim_vars['vz'] = np.zeros(shape)
        micro_model.prim_vars['n'] = np.zeros(shape)
        micro_model.prim_vars['p'] = np.zeros(shape)

        args_for_pool=[]
        for h, T in enumerate(micro_model.domain_vars['t']):
            for i, X in enumerate(micro_model.domain_vars['x']):
                for j, Y in enumerate(micro_model.domain_vars['y']):
                    for k, Z in enumerate(micro_model.domain_vars['z']):
                        point, idxs = [T,X,Y,Z], [h,i,j,k]
                        args_for_pool.append((point, idxs))

        init = Aenus3D_h5py.RID_initializer
        initargs=(enclosed_grid, [r,phi,z], rho, Pgas, br, bphi, bz, vr, vphi, vz, method)

        with mp.Pool(initializer=init, initargs=initargs, processes=n_cpus) as pool:
            print('Interpolating on the Cartesian grid in parallel with {} processes'.format(pool._processes), flush=True)
            for result in pool.starmap(Aenus3D_h5py.task_RID, args_for_pool):
                h,i,j,k = result[0]
                micro_model.prim_vars['n'] = result[1]
                micro_model.prim_vars['P'] = result[2]
                micro_model.prim_vars['Bx'] = result[3]
                micro_model.prim_vars['By'] = result[4]
                micro_model.prim_vars['Bz'] = result[5]
                micro_model.prim_vars['vx'] = result[6]
                micro_model.prim_vars['vy'] = result[7]
                micro_model.prim_vars['vz'] = result[8]

        # Setting up the auxiliary variables: Lorentz factor 
        W = (micro_model.prim_vars['vx']**2 + micro_model.prim_vars['vy']**2 + micro_model.prim_vars['vz']**2)
        W = 1. / np.sqrt(1-W)
        micro_model.aux_vars['W'] = W

        # Setting up the enthalpy: need to take care of dimensionality of the polytropic constant. 
        polytrope_K = eos_para['polytrope_K']
        Gamma_b = eos_para['Gamma_b']
        Gamma_th = eos_para['Gamma_th']
        Pth = (micro_model.prim_vars['p'] / self.pressure_factor) - polytrope_K * np.power(micro_model.prim_vars['n'] * self.density_factor, Gamma_b) #thermal pressure in cgs
        Pth = Pth * self.pressure_factor # thermal pressure in geometrized units
        Pb = micro_model.prim_vars['p'] - Pth
        h = micro_model.prim_vars['p'] / micro_model.prim_vars['n'] + Pth / (Gamma_th-1) + Pb/(Gamma_b-1)
        micro_model.aux_vars['h'] = h  

        micro_model.aux_vars.update(eos_para)

