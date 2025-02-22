# LAGRANGIAN FILTERING 

This is the first __fully-covariant lagrangian filtering scheme__ for applications to relativistic turbulence. 
It has been developed by Dr. Thomas Celora, Mr. Marcus J. Hatton and Dr. Ian Hawke. The codebase is meant to be used to investigate models of relativistic turbulent flows, and calibrate sugbgrid closure schemes for large-eddy simulations in a fully-covariant fashion. As such, results obtained within this framework may be lifted into an arbitrary, curved spacetime. This makes it a useful tool for studying, for example, binary neutron star mergers via numerical relativity simulations.  

Related publications: 
* [Fibration framework](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.084090) 
* [Lagrangian filtering](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.123040)
* [A higher-level strategy](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.123039)


# GETTING STARTED 
## HOW TO INSTALL IT
To begin using the code, first clone this repository
```
git clone https://github.com/Lagrangian-filtering/Lagrangian-filtering
```
The code is entirely developed in the Python language, so a normal Python interpreter plus some additional libraries will suffice. The complete list of required libraries/packages can be found in `requirements.txt`. You should also have data (in HDF5 format) produced by a numerical box simulation of hydrodynamics turbulence. For example, you may get this by running one of the examples of the [METHOD](https://github.com/AlexJamesWright/METHOD/tree/master) codebase. More information on how such data should be stored for running it through the pipeline can be found in [Lagrangian filtering](https://arxiv.org/abs/2405.13593).

## HOW TO USE IT
The code is developed in a pipeline-like fashion. The main classes are in the `master_files` folder: classes for representing simulations' and filtered data, filtering classes and classes for analyising and calibrating a closure scheme. 

As an example, `filter_scripts` contains scripts for running simulation data through the filtering pipeline and visualizing the outcome. 
The configuration file `config_filter.txt` should be modified accordingly to set the relevant parameters for the simulations. 
The key script in here is `pickling_meso.py`, which can be used to run simulation data through the entire pipeline and store the results in compact binary representation (pickle format). The folder also contains example submit script to launch the simulation in a cluster. 

The folder `calibration_scripts` contains scripts for calibrating and comparing subgrid models. In particular there are scripts for visualizing correlations, performing regressions and checking the performance of the a-priori tests. This folder is structured in a similar fashion as `filter_scripts`: it contains a single configuration file, `config_calibration.txt` and example submit scripts. 
Each script begins with a short description of what it is intended to do. 

# AUTHORS
* [Dr. Thomas Celora](https://www.researchgate.net/profile/Thomas-Celora)
* [Mr. Marcus Hatton](https://www.southampton.ac.uk/people/5y8l7z/mr-marcus-hatton)
* [Dr. Ian Hawke](https://www.southampton.ac.uk/people/5x29mr/doctor-ian-hawke)