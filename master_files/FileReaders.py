# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:00:00 2023

@authors: Marcus & Thomas
"""

import h5py
import glob
import numpy as np

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

    def read_in_data3D(self, micro_model, crop=None, stride=1):   
        """
        Store 3+1 dimensional data from files into micro_model

        Parameters
        ----------
        micro_model: instance of a MicroModel 
            strs in micromodel have to be the same as hdf5 files output from METHOD.

        crop: None or ((i0, i1), (j0, j1), (k0, k1))
            index ranges (python slice semantics, stop exclusive; None for an
            axis end keeps that end) of the cells to keep along x, y, z.
            Only the retained region is read from disk. The domain variables
            (xmin, xmax, nx, x, ...) are rewritten to describe the retained
            region. Note a crop is not periodic: derivatives are one-sided at
            its faces, so keep a margin around the region of interest.

        stride: int
            keep every stride-th cell along each axis (after cropping).
            dx, dy, dz are multiplied by stride; dt is unchanged.

        Notes 
        -----
            use and adapt translating dictionary if you want to store data in micromodel 
            using different keys than those from METHOD.

        """ 

        # Region to read: slices into the (nx, ny, nz) arrays, resolved against the file shape
        if int(stride) < 1:
            raise ValueError(f'stride must be a positive integer, got {stride}')
        stride = int(stride)
        full_shape = self.hdf5_files[0]['Primitive/rho'].shape
        if crop is None:
            crop = ((None, None), (None, None), (None, None))
        if len(crop) != 3:
            raise ValueError('crop must have one (start, stop) pair per axis x, y, z')
        ranges = []
        for axis_range, n in zip(crop, full_shape):
            start, stop, step = slice(axis_range[0], axis_range[1], stride).indices(n)
            if len(range(start, stop, step)) == 0:
                raise ValueError(f'crop {axis_range} selects no cells on an axis of length {n}')
            ranges.append((start, stop, step))
        sl = tuple(slice(*r) for r in ranges)
        region_changed = (crop != ((None, None), (None, None), (None, None))) or stride != 1

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
                    micro_model.prim_vars[prim_var_str].append( self.hdf5_files[counter]["Primitive/"+method_str][sl] )

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
                    micro_model.aux_vars[aux_var_str].append( self.hdf5_files[counter]["Auxiliary/"+method_str][sl] )
                micro_model.aux_vars[aux_var_str] = np.array(micro_model.aux_vars[aux_var_str])
            except KeyError:
                print(f'{method_str} is not in the hdf5 dataset: check Auxiliary/')
 
        # As METHOD saves endTime, the time variables (and points) need to be dealt with separately
        for dom_var_str in micro_model.domain_int_strs: 
            try: 
                if dom_var_str == 'nt': 
                    pass
                else: 
                    micro_model.domain_vars[dom_var_str] = int(np.ravel(self.hdf5_files[0]['Domain'].attrs[dom_var_str])[0])
            except KeyError:
                print(f'{dom_var_str} is not in the hdf5 Domain group attributes')

        for dom_var_str in micro_model.domain_float_strs:
            try:
                if dom_var_str in ['tmin', 'tmax']:
                    pass
                else:
                    micro_model.domain_vars[dom_var_str] = float(np.ravel(self.hdf5_files[0]['Domain'].attrs[dom_var_str])[0])
            except KeyError:
                print(f'{dom_var_str} is not in the hdf5 Domain group attributes')

        for dom_var_str in micro_model.domain_array_strs:
            try:
                if dom_var_str in ['t','points', 'x', 'y', 'z']:
                    pass
                else:
                    micro_model.domain_vars[dom_var_str] = self.hdf5_files[0]['Domain/' + dom_var_str][:]
            except KeyError:
                print(f'{dom_var_str} is not in the hdf5 dataset: check Domain/')


        # Re-express the domain in terms of the retained cells (nothing changes for a full read)
        if region_changed:
            for axis, (start, stop, step) in zip('xyz', ranges):
                d_old = micro_model.domain_vars['d' + axis]
                micro_model.domain_vars[axis + 'min'] += start * d_old
                micro_model.domain_vars['d' + axis] = d_old * step
                micro_model.domain_vars['n' + axis] = len(range(start, stop, step))
                micro_model.domain_vars[axis + 'max'] = (micro_model.domain_vars[axis + 'min']
                    + micro_model.domain_vars['n' + axis] * micro_model.domain_vars['d' + axis])

        micro_model.domain_vars['nt'] = self.num_files
        for counter in range(self.num_files):
            micro_model.domain_vars['t'].append(float(np.ravel(self.hdf5_files[counter]['Domain'].attrs['endTime'])[0]))

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