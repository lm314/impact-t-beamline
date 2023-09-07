import os
import subprocess
import logging
import shutil
import numpy as np
import yaml
import inspect
import pandas as pd

from beamline_configuration import BeamlineConfiguration
import pyPartAnalysis.read as rd

with open(os.path.join(os.getcwd(),f'config.yaml'), 'r') as file:
    yaml_file = yaml.safe_load(file)
    IMPACT_EXE_PATH = yaml_file.get('IMPACT_EXE_PATH','')
    DATA_DIR = yaml_file.get('DATA_DIR','')
    
def gaussian_FWHM_to_RMS(FWHM):
    return FWHM/(2*np.sqrt(2*np.log(2))).item()

def gaussian_RMS_to_FWHM(RMS):
    return RMS*(2*np.sqrt(2*np.log(2))).item()

def get_default_args(func):
    # returns default arguments from a function
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def try_except_timeout(func):
    # wrapper to handle cases where an ImpactT_Beamline times out.
    # returns None when run times out.
    # user can specify the value return by a timeout using the keyword argument
    # "timeout_value" in the function this wraps
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except subprocess.TimeoutExpired:
            print('Command timed out')
            kwds = get_default_args(func)
            kwds.update(kwargs)
            if 'timeout_value' in kwds.keys():
                return kwds['timeout_value']
            else:
                return None;
        except subprocess.CalledProcessError as e:
            print('Command failed with error:', e.returncode, e.output)  
            raise
    wrapper.__signature__ = inspect.signature(func)
    return wrapper

def block_negative_velocity(func):
    # returns None if the ref particle z velocity becomes negative
    def wrapper(*args, **kwargs):
        # Get information about the arguments accepted by func
        argspec = inspect.getfullargspec(func)
        arg_names = argspec.args
        
        # Filter positional arguments based on their type
        myclass_args = [arg for arg in args if isinstance(arg, ImpactTBeamline)]
        
        # Filter keyword arguments based on their type
        myclass_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, ImpactTBeamline)}
        
        # Print the MyClass arguments
        temp_beamline = myclass_args[0]
        z_df = temp_beamline.getFort(fort_num=26)
        
        if np.any(z_df.avgPz<0):
            kwds = get_default_args(func)
            kwds.update(kwargs)
            if 'negative_v_value' in kwds.keys():
                return kwds['negative_v_value']
            else:
                return None;
        else:
            return func(*args, **kwargs)
    # preserve arguments of func when chaining decorators
    wrapper.__signature__ = inspect.signature(func)
    return wrapper

class ImpactTBeamline:
    def __init__ (self,settings,impact_file,gen=None,timeout=None,num_process=1,impact_exe_path=IMPACT_EXE_PATH,run_dir=None,data_files=None,has_particle_id=True):
        # settings is the dictionary output by the BeamlineConfiguration.gen method
        # gen is a distgen Generator
        self.settings = settings
        self.impact_file = impact_file
        self.gen = gen
        self.timeout = timeout
        self.num_process = num_process
        self.impact_exe_path = impact_exe_path
        self.data_files = data_files
        if run_dir is None:
            self.run_dir = os.getcwd()
        else:
            self.run_dir = run_dir
        self.has_particle_id = has_particle_id
    
    def callImpactT(self):
        # calls IMPACT-T executable via mpirun
        temp = subprocess.check_call(f"mpirun -n {self.num_process} {self.impact_exe_path} > output.txt",
                                      shell=True,
                                      cwd=self.run_dir,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL,
                                      timeout=self.timeout) 
                # temp = subprocess.check_call(f"mpirun -n {self.num_process} -c {self.num_process} {self.impact_exe_path} > output.txt",
                #                       shell=True,
                #                       cwd=self.run_dir,
                #                       timeout=self.timeout) 
    
    def makeImpactIn(self,impact_file_name = 'ImpactT.in'):
        # generates an ImpactT.in file with the values substituted in for the variables
        impact_edit = self.impact_file.replace(variables=self.get_ImpactTin_settings())
        impact_edit.write(filename=os.path.join(self.run_dir,impact_file_name))
    
    def populateDataFiles(self):
        # copy the files needed to run the ImpactT.in file, e.g. rfdata1 or partcl.data
        for file in self.data_files:
            shutil.copy(os.path.join(DATA_DIR,file),self.run_dir)   
    
    def make_run_dir(self):
        # make the directory specified by self.run_dir
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
    
    def run(self):
        # runs IMPACT-T case, including making the directory, ImpactT.in, 
        # distribution, copying over the data files, and calling the executable
        self.make_run_dir()
        self.makeImpactIn()
        self.makeDist()
        self.populateDataFiles()
        self.callImpactT()
   
    def makeDist(self,file_name='partcl.data'):
        # generate a distribution file using distgen Generator of self.gen
        if self.gen is not None:
            distgen_settings = self.get_distgen_settings()

            for key,val in distgen_settings.items():
                self.gen[key] = val
            pg = self.gen.run()
            pg.drift_to_t(pg['mean_t'])
            pg.write_impact(os.path.join(self.run_dir,file_name),dev_branch=self.has_particle_id)
    
    def getFort(self,fort_num):
        # reads summary IMPACT-T file into a pandas DataFrame, i.e. not distribution files
        file_name = os.path.join(self.run_dir,f'fort.{fort_num}')
        return rd.read_fort_t(file_name)
    
    def getDist(self,fort_num):
        # reads an IMPACT-T distribution file into a pandas DataFrame
        file_name = os.path.join(self.run_dir,f'fort.{fort_num}')
        return rd.read_GB(file_name)
        
    def getFort_z_pos(self,fort_num,z_pos_list):
        # gets the pandas DataFrame rows with the z positions closest to z_pos_list
        df = self.getFort(fort_num)
        
        #fort 18 uses dist instead of z
        if fort_num == 18:
            z = df['dist']
        else:
            z = df['z']
    
        result = []
        for z_pos in z_pos_list:
            result.append(df.iloc[np.argmin(np.abs(z - z_pos))])
        
        return pd.DataFrame(result)
    
    def get_distgen_settings(self):
        # get variables intented for the distgen file
            settings_combi = BeamlineConfiguration.split(self.settings)
            return settings_combi.get('distgen',settings_combi)
            
    def get_ImpactTin_settings(self):
        # get variables intented for the ImpactTin file
        settings_combi = BeamlineConfiguration.split(self.settings)
        return settings_combi.get('original',settings_combi)        
