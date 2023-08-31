# impact-t-beamline
Python class for creating ImpactT.in files and running them from the command line

# config.yaml

In order for the executable to be called correctly and the requisite data files to be copied to the run directory, a config.yaml file must be included in the directory the python program is run from. It should define the following:

```yaml
IMPACT_EXE_PATH: '/scratch/$USER/impact_t_exe/ImpactTexe-mpi-nvhpc'
DATA_DIR: '/scratch/$USER/Shorter_CXLS/tuning_tools/rf_phasing_4_linac/CXLS_impact_files'
```


