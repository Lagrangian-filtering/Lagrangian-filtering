# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:21:02 2022

@authors: Marcus & Thomas
"""

# from multiprocessing import Process, Pool
import numpy as np
# from timeit import default_timer as timer
import cProfile, pstats, io
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
from scipy import fft

class Base(object):
    """
    Base class containing basic routines that are used by multiple 
    other classes. 
    """

    @staticmethod
    def Mink_dot(vec1, vec2):
        """
        Returns the dot product (Minkowski metric) of input vectors 'vec1', 'vec2'

        Parameters:
        -----------
        vec1, vec2 : list of floats (or np.arrays)

        Return:
        -------
        mink-dot (cartesian) in 1+n dim
        """
        if len(vec1) != len(vec2):
            print("The two vectors passed to Mink_dot are not of same dimension!")

        dot = -vec1[0]*vec2[0]
        for i in range(1,len(vec1)):
            dot += vec1[i] * vec2[i]
        return dot
  
    @staticmethod
    def get_rel_vel(spatial_vels):
        """
        Build unit vectors starting from spatial components
        Needed as this will enter the minimization procedure

        Parameters:
        ----------
        spatial_vels: list of floats

        Returns:
        --------
        list of floats: the d+1 vector, normalized wrt Mink metric
        """
        W = 1 / np.sqrt(1-np.sum(spatial_vels**2))
        return W * np.insert(spatial_vels,0,1.0)

    """
    A pair of functions that work in conjuction (thank you stack overflow).
    find_nearest returns the closest value to 'value' in 'array',
    find_nearest_cell then takes this closest value and returns its indices.
    Should now work for any dimensional data.
    """
    @staticmethod
    def find_nearest(array, value):
        """
        Returns closest value to input 'value' in 'array'

        Parameters: 
        -----------
        array: np.array of shape (n,)

        value: float

        Returns:
        --------
        float 

        Note:
        -----
        To be used together with find_nearest_cell.  
        """
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1]
        else:
            return array[idx]
        
    @staticmethod    
    def find_nearest_cell(point, points):
        """
        Use find nearest to find closest value in a list of input 'points' to 
        input 'point'. 

        Parameters:
        -----------
        point: list of d+1 float

        points: list of lists of d+1 floats 

        Returns:
        --------
        List of d+1 indices corresponding to closest value to point in points
        """
        if len(points) != len(point):
            print("find_nearest_cell: The length of the coordinate vector\
                   does not match the length of the coordinates.")
        positions = []
        for dim in range(len(point)):
            positions.append(Base.find_nearest(points[dim], point[dim]))
        return [np.where(points[i] == positions[i])[0][0] for i in range(len(positions))]

    @staticmethod
    def get1DFourierTrans(u, nx, ny):
        """
        Returns the 1D discrete fourier transform of the variable u along both 
        the x and y directions ready for the power spectrum method.
        Parameters
        ----------
        u : ndarray
            Two dimensional array of the variable we want the power spectrum of
        Returns
        -------
        uhat : array (N,)
            Fourier transform of u
        """
       
        NN = nx // 2
        uhat_x = np.zeros((NN, ny), dtype=np.complex_)
    
        for k in range(NN):
            for y in range(ny):
                # Sum over all x adding to uhat
                for i in range(nx):
                    uhat_x[k, y] += u[i, y] * np.exp(-(2*np.pi*1j*k*i)/nx)

        NN = ny // 2
        uhat_y = np.zeros((NN, nx), dtype=np.complex_)
        
        for k in range(NN):
            for x in range(nx):
                # Sum over all y adding to uhat
                for i in range(ny):
                    uhat_y[k, x] += u[x, i] * np.exp(-(2*np.pi*1j*k*i)/ny)

        return (uhat_x / nx), (uhat_y / ny) 
    
    @staticmethod
    def getPowerSpectrumSq(u, nx, ny, dx, dy):
        """
        Returns the integrated power spectrum of the variable u, up to the Nyquist frequency = nx//2
        Parameters
        ----------
        u : ndarray
            Two dimensional array of the variable we want the power spectrum of
        """
        uhat_x, uhat_y = Base.get1DFourierTrans(u, nx, ny)

        NN = nx // 2
        P_x = np.zeros(NN)
    
        for k in range(NN):
            for j in range(ny):
                P_x[k] += (np.absolute(uhat_x[k, j])**2) * dy
        P_x = P_x / np.sum(P_x)

        NN = ny // 2
        P_y = np.zeros(NN)

        for k in range(NN):
            for i in range(nx):
                P_y[k] += (np.absolute(uhat_y[k, i])**2) * dx
        P_y = P_y / np.sum(P_y)

        return [P_x, P_y]
    
    def profile(self, fnc):
        """A decorator that uses cProfile to profile a function"""
        def inner(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            retval = fnc(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            return retval
        return inner

    @staticmethod
    def get1DFourierTrans_smart(u):
        """
        Returns the 1D discrete fourier transform of the variable u along 
        each direction ready for the power spectrum method.

        Parameters
        ----------
        u : ndarray
            the variable we want the power spectrum of

        Returns
        -------
        uhat : array (N,)
            Fourier transform of u, where N is len(u.shape)

        Notes
        ------
        Way slower than the inbuilt routines in scipy 
        """
        dims = len(u.shape)
        NNs = []
        for d in range(dims):
            NNs.append(u.shape[d])

        uhats = []

        for d in range(dims):
            nns = list(NNs)
            nns[d] = int(nns[d]/2) + 1

            uhat = np.zeros(tuple(nns), dtype=np.complex_)

            for idx in np.ndindex(uhat.shape):
                k = idx[d]
                remainig_idxs = list(idx)
                del remainig_idxs[d]
                for i in range(NNs[d]): 
                    real_space_idx = list(remainig_idxs)
                    real_space_idx.insert(d, i)
                    
                    uhat[idx] += u[tuple(real_space_idx)] * np.exp(-(2*np.pi*1j*k*i)/NNs[d])
            # uhat = uhat/ NNs[d]
            uhats.append(uhat)

        return uhats 

    @staticmethod
    def get1DPowerSpectrumSq_smart(u, Ds):
        """
        Routine to get the 1D integrated spectra of the input var 'u'. 
        For e.g. the x-direction: compute the fourier series along the x direction only, 
        then integrate (the modulus square) along the remaining spatial directions. 
        Repeat for all the spatial dimensions associated with input 'u'

        Works in any dimension.

        Parameters
        -----------
        u: ndarray,
            the data to compute the power spectra of
        
        Ds: list
            real space grid increments in each direction, must be compatible with u

        Returns
        -------
        ndarray of shape (N,) where N is the len(u.shape)

        Notes
        ------
        Useful for checking if there is a significant difference in the 1d spectra
        """
        dims = len(u.shape)
        if len(Ds) != dims:
            print("Data and increments are not compatible")
            return None
        
        NNs = []
        for d in range(dims):
            NNs.append(u.shape[d])

        # uhats = get1DFourierTrans_smart(u)
        uhats = []
        for i in range(dims):
            uhat = fft.rfft(u, axis=i, overwrite_x=False)
            uhats.append(uhat)

            
        PS = []
        for d in range(dims):
            nn = int(NNs[d]/2)
            ps = np.zeros(nn + 1)

            ds = list(Ds)
            del ds[d]
            increment = 1. 
            for i in range(len(ds)):
                increment *= float(ds[i])

            uhat = uhats[d]
            for idx in np.ndindex(uhat.shape):
                k = int(idx[d])
                ps[k] += (np.absolute(uhat[idx]) ** 2) * increment

            ps = ps / np.sum(ps)
            PS.append(ps)
        return PS

    @staticmethod
    def Spectrum3D_annular_avg(u, dx, dy, dz):
        """
        Compute the 3D FFT of u. Then, assuming isotropy work out the spectrum 
        as a function of the wavenumber by integrating over annular ring in 
        frequency space. Normalization is fixed to make sure the result is an 
        energy-like spectrum.

        As the normalization is specific to the number of dimensions, this routine 
        is meant to be used on 3D gridded data only.  

        Parameters
        ----------
        u: ndarray
            the gridded 3D data you want to take the derivative of

        dx, dy, dz: float
            the grid spacings in each direction 

        Returns
        -------
        ndarray: the spectra averaged over the annular rings in frequency space

        Notes
        -----
        """
        Nx, Ny, Nz = u.shape
        Nmax = math.floor( np.amax([Nx/2, Ny/2, Nz/2]) * np.sqrt(3) )

        Lx, Ly, Lz = dx * Nx , dy * Ny , dz * Nz
        
        dk = 2 * np.pi / (np.min([Lx, Ly, Lz]))
        
        # computing the mode labels corresponding to each annular ring in frequency space
        Rps = []
        lens = []
        for p in range(math.floor(Nmax)):
            rp = []
            for l in range(math.floor(Nx/2)):
                for m in range(math.floor(Ny/2)):
                    for n in range(math.floor(Nz/2)):
                        kx, ky, kz = l * 2 * np.pi / Lx , m * 2 * np.pi / Ly , n * 2 * np.pi / Lz 
                        ksq = np.sqrt(kx**2 + ky**2 + kz**2)
                        if ksq >= dk * p and ksq < (p+1)* dk:  
                            rp.append([l,m,n])
            
            lens.append(len(rp))
            Rps.append(rp)

        uhat = fft.fftn(u, overwrite_x=False)
        uhat = uhat[:math.ceil(Nx/2), :math.ceil(Ny/2), :math.ceil(Nz/2)]

        # summing over the annular rings labels
        uhat_mod = np.zeros((Nmax,))

        for kp in range(len(uhat_mod)):
            relevant_wnums = Rps[kp]
            for elem in relevant_wnums: 
                uhat_mod[kp] += np.absolute(uhat[tuple(elem)])**2
            
            kmodsq = ((kp+1) * dk)**2 
            uhat_mod[kp] = kmodsq * uhat_mod[kp] 

            Np = len(relevant_wnums)
            if Np != 0:
                uhat_mod[kp] = uhat_mod[kp] / Np

        # normalizing
        prefactor = (Lx * Ly * Lz) / (Nx * Ny * Nz)**2 
        prefactor = (prefactor * 4 * np.pi) / (2* np.pi)**3 

        return prefactor * uhat_mod

class MySymLogPlotting(object):
    """
    Class containing useful routines for plotting using a symmetric logarithmic norm and a divergent colormap. 
    In particular, we want to make sure the norm is set-up in such a way that the midvalue is at the center
    of a divergent colormap. The scaling to above/below the mid-value is different to ensure this.  
    """
    @staticmethod
    def symlog_num(num):
        """
        Return the symlog of a number

        Parameters:
        -----------
        num: float

        Returns:
        --------
        The symlog of the input num (separate copy)
        """
        if np.abs(num) +1. == 1.0:
            result = num
        else:
            result = np.sign(num) * np.log10(np.abs(num)+1.)
        return result

    @staticmethod
    def inverse_symlog_num(num):
        if num > 0:
            return 1 + 10 ** num
        elif num < 0:
            return 1- 10 ** (- num)
        else:
            return 0

    @staticmethod
    def symlog_var(var):
        """
        Return the symlog of an array.

        Parameters:
        -----------
        var: np.array of any shape

        Returns:
        --------
        The symlog of the input var (separate copy)
        """
        count_zeros=0
        temp = np.empty_like(var)
        for index in np.ndindex(var.shape):
            value = var[index]
            if value == 0: 
                count_zeros +=1
            else: 
                temp[index] = MySymLogPlotting.symlog_num(value)
        if count_zeros >= 1:
            print('Careful: there are {} zeros in the data'.format(count_zeros))
        return temp

    @staticmethod
    def get_mysymlog_var_ticks(var):
        """
        Method to automatize the computation of the ticks and nodes for a variable.
        Nodes are then to be used within the class MyThreeNodesNorm. 
        Ticks and labels are for the colorbar of the plot of input 'var'. 

        Parameters: 
        -----------
        var: np.array
            This HAS TO take both positive and negative values 

        Returns:
        --------
        ticks: list
            list of tick points to be used by the colorbar

        ticks_labels: list
            list of corresponding labels for the colorbar

        nodes: list of len=5
            the extrame and the three central nodes to be used by MyThreeNodesNorm

        Notes:
        ------
        The ticks/nodes and labels are computed like this: start from negative values, identify min 
        and max values of the negative part of input 'var' to identify relevant ticks and nodes. 
        Then add a zero (tick and node) and proceed to the positive values.
        """
        ticks = []
        ticks_labels = []
        nodes = []
    

        pos_var = np.ma.masked_where(var <0., var, copy=True).compressed()
        pos_var_small = np.ma.masked_where(pos_var >=1., pos_var, copy=True).compressed()
        pos_var_large = np.ma.masked_where(pos_var <1., pos_var, copy=True).compressed()

        neg_var = np.ma.masked_where(var >0., var, copy=True).compressed()
        neg_var_small = np.ma.masked_where(neg_var <=-1., neg_var, copy=True).compressed()
        neg_var_large = np.ma.masked_where(neg_var >-1., neg_var, copy=True).compressed()

        # Working out nodes, ticks and ticks_labels for the negative range
        if len(neg_var_large) >0: 
            # print('There are negative large values', flush=True)
            vmin = np.amin(neg_var_large)
            new_nodes = [MySymLogPlotting.symlog_num(vmin)]
            new_ticks = new_nodes
            new_ticks_labels = [r'$-10^{%d}$'%(int(np.log10(-vmin)))]
            
            nodes += new_nodes
            ticks += new_ticks
            ticks_labels += new_ticks_labels
            
            
            if len(neg_var_small) == 0: 
                # print('Actually: only negative large values', flush=True)
                vmax = np.amax(neg_var_large)
                new_nodes = [MySymLogPlotting.symlog_num(vmax)]
                new_ticks = new_nodes
                new_ticks_labels = [r'$-10^{%d}$'%(int(np.log10(-vmax)))]

                nodes += new_nodes
                ticks += new_ticks
                ticks_labels += new_ticks_labels

            else:
                # print('And also negative small values', flush=True)
                vmin = np.amin(neg_var_small)
                vmax = np.amax(neg_var_small)

                new_nodes = [MySymLogPlotting.symlog_num(vmax)]
                # new_ticks = [symlog_num(vmin), symlog_num(vmax)]
                new_ticks = [MySymLogPlotting.symlog_num(vmax)]
                # new_ticks_labels = [r'$-10^{%d}$'%(int(d)) for d in np.log10([-vmin,-vmax])]
                new_ticks_labels = [r'$-10^{%d}$'%(int(d)) for d in np.log10([-vmax])]
                
                ticks += new_ticks
                ticks_labels += new_ticks_labels
                nodes += new_nodes
                
        else: # len(neg_var_large)==0:
            # print('Only negative small values', flush=True)
            vmin = np.amin(neg_var_small)
            vmax = np.amax(neg_var_small)

            # print(vmin, vmax, "\n")
            new_nodes = [MySymLogPlotting.symlog_num(vmin), MySymLogPlotting.symlog_num(vmax)]
            # print(new_nodes)
            new_ticks = new_nodes
            new_ticks_labels = [r'$-10^{%d}$'%(int(d)) for d in np.log10([-vmin,-vmax])]

            ticks += new_ticks
            ticks_labels += new_ticks_labels
            nodes += new_nodes


        nodes += [0.]
        ticks += [0.]
        ticks_labels += ['0']

        # Working out the remaining nodes, ticks and ticks_labels for the positive range

        if len(pos_var_large)==0:
            # print('Only positive small values', flush=True)
            vmin = np.amin(pos_var_small)
            vmax = np.amax(pos_var_small)

            # print(vmin, vmax)
            new_nodes = [MySymLogPlotting.symlog_num(vmin), MySymLogPlotting.symlog_num(vmax)]
            # print(new_nodes)
            new_ticks = new_nodes
            new_ticks_labels = [r'$10^{%d}$'%(int(d)) for d in np.log10([vmin,vmax])]
    
            ticks += new_ticks
            ticks_labels += new_ticks_labels
            nodes += new_nodes

        else: # len(pos_var_large) > 0: 
            # print('There are positive large values', flush=True)

            if len(pos_var_small) >0:
                # print('And also positive small values', flush=True)
                vmin = np.amin(pos_var_small)
                vmax = np.amax(pos_var_small)

                # new_nodes = [symlog_num(vmax)]
                new_nodes = [MySymLogPlotting.symlog_num(vmin)]
                # new_ticks = [symlog_num(vmin), symlog_num(vmax)]
                new_ticks = [MySymLogPlotting.symlog_num(vmin)]
                # new_ticks_labels = [r'$10^{%d}$'%(int(d)) for d in np.log10([vmin,vmax])]
                new_ticks_labels = [r'$10^{%d}$'%(int(d)) for d in np.log10([vmin])]
                
                ticks += new_ticks
                ticks_labels += new_ticks_labels
                nodes += new_nodes

                vmax = np.amax(pos_var_large)
                new_nodes = [MySymLogPlotting.symlog_num(vmax)]
                new_ticks = new_nodes
                new_ticks_labels = [r'$10^{%d}$'%(int(np.log10(vmax)))]
                
                ticks += new_ticks
                ticks_labels += new_ticks_labels
                nodes += new_nodes
                

            else: # len(pos_var_small) ==0:
                vmin = np.amin(pos_var_large)
                vmax = np.amax(pos_var_large)

                new_nodes = [MySymLogPlotting.symlog_num(vmin), MySymLogPlotting.symlog_num(vmax)]
                new_ticks = new_nodes
                new_ticks_labels = [r'$10^{%d}$'%(int(d)) for d in np.log10([vmin,vmax])]

                ticks += new_ticks
                ticks_labels += new_ticks_labels
                nodes += new_nodes

        return ticks, ticks_labels, nodes

class MyThreeNodesNorm(mpl.colors.Normalize):
    """
    Sub-classing colors.Normalize: the norm has three inner nodes plus the extrema.
    Within each segment (delimited by a node or extrema) you have linear interpolation. 
    
    Should be used when plotting quantities that are both positive and negative, and you
    want to highlight 1) where a critical value (middle_node) is 2) the closest values some 
    variable takes to its left and right
    """
    def __init__(self, nodes, clip=False):
        """
        Parameters: 

        nodes: array of five numbers in strictly ascending order (the nodes)
        """
        if len(nodes)!=5: 
            raise ValueError('The class MyThreeNodesNorm requires 5 nodes: the extrema and the three central')

        for i in range(len(nodes)-1):
            if nodes[i+1] <=nodes[i]:
                raise ValueError('nodes must be in monotonically ascending order!')
                
        super().__init__(nodes[0], nodes[4], clip)
        self.first_node = nodes[1]
        self.central_node = nodes[2]
        self.third_node = nodes[3]

    def __call__(self, value, clip=None):
        x = [self.vmin, self.first_node, self.central_node, self.third_node, self.vmax]
        y = [0, 0.4, 0.5, 0.6, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y = [self.vmin, self.first_node, self.central_node, self.third_node, self.vmax]
        x = [0, 0.4, 0.5, 0.6, 1.]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)
