# LAGRANGIAN FILTERING 

This is the first __fully covariant filtering scheme__ applied to relativistic hydrodynamic turbulence, as developed by Dr. Thomas Celora, Mr. Marcus J. Hatton and Dr. Ian Hawke. It is meant to be used to investigate models of relativistic turbulent flows, and calibrate sugbgrid models in a covariant fashion. In particular, the covariance of our approach means that results obtained within this framework may be lifted into an arbitrary, curved spacetime. This makes it well-suited for studies of, for example, binary neutron star mergers via numerical relativity simulations.  

Related publications: 
* [Fibration framework](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.084090) 
* [Lagrangian filtering]()
* [A higher-level strategy]()

# GETTING STARTED 
## HOW TO INSTALL IT
To begin, first clone this code repository
```
git clone https://github.com/Lagrangian-filtering/Lagrangian-filtering
```
Should also have data (in HDF5 format) of a numerical box simulation. For example, you may get this by running one of the examples of the [METHOD](https://github.com/AlexJamesWright/METHOD/tree/master) codebase. 

## HOW TO USE IT
The code is developed in a pipeline-like fashion. The main classes are in the master_files folder: classes for representing simulations' and filtered data, filtering classes and classes for analyising and calibrating a closure scheme. 

As an example, filter_scripts contains scripts for running through the filtering pipeline and visualizing the outcome. 
The configuration file config_filter.txt should be modified accordingly to set the relevant parameters for the simulations. 
The key script in here is `pickling_meso.py`, which can be used to run simulation data through the entire pipeline and store the results in compact binary representation (pickle format). The folder also contains example submit script to launch the simulation in a cluster. 

The folder calibration_scripts contains scripts for calibrating and comparing subgrid models. In particular there are scripts for visualizing correlations, performing regressions and checking the performance of the a-priori tests. 

# AUTHORS
* [Dr. Thomas Celora](https://www.ice.csic.es/about-us/staff)
* [Mr. Marcus Hatton](https://www.southampton.ac.uk/people/5y8l7z/mr-marcus-hatton)
* [Dr. Ian Hawke](https://www.southampton.ac.uk/people/5x29mr/doctor-ian-hawke)