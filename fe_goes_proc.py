# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:36:21 2019

Processing GOES-R satelliate imagery using satpy package
note: currently, it's only designed to process a specific area at SS

@author: ybcheng
"""

import os
import satpy
import glob
import numpy as np
from satpy import Scene
from pyresample import geometry
from pyresample.geometry import AreaDefinition


def set_params():
    """
    Definitions of camera parameters related to radiometric calibration.
    Note: the dictionary can be revised as system-based, e.g. replace
    'ids' with 'pomona2', replace '611' '612' with '0' '1'
    """

    goes = {
            "rgb":
                {"b1": "C01",
                 "b2": "C02",
                 "b3": "C03"},
            "nclr":
                {"b1": "C02",
                 "b2": "C03",
                 "b3": "C05"},
            "fclr":           
                {"b1": "C01",
                 "b2": "C03",
                 "b3": "C05"},
            "dust":  
                {"b1": "C11",
                 "b2": "C13",
                 "b3": "C15"}             
            }    
        
    return locals() 


def check_data(in_dir, im_type='rgb'):
    """
    check imagery data in the in_dir and make sure all the three bands are
    there to make either "true-color" or "false-color" or "dust-rgb" composite
    for rgb , check: C01(0.47um), C02(0.64um), C03(0.86um)
    for fclr, check: C05(1.60um), C03(0.86um), C01(0.47um)
    for dust, check: C11(8.40um), C13(10.3um), C15(12.3um)
    
    Parameters:
    ----
    in_dir: str
        the folder where GOES-R images are
    im_type: str
        'rgb' or 'fclr' or 'dust'
    """ 
    
    
    if im_type is 'rgb':
        num01 = len(glob.glob(in_dir + '*-M*C01_*.nc'))
        num02 = len(glob.glob(in_dir + '*-M*C02_*.nc'))
        num03 = len(glob.glob(in_dir + '*-M*C03_*.nc'))
        if (num01 != num02) or (num01 != num03) or (num02 != num03):
            print('check database')
            print('C01: %i files' % num01 )
            print('C02: %i files' % num02 )
            print('C03: %i files' % num03 )
            dt_list=[]
        else:
            print('database looks good')
            flist = [os.path.basename(x) for x in glob.glob(in_dir + '*-M*C01_*.nc')]
            dt_list = [y.split('_')[3][1:12] for y in flist]
    elif im_type is 'nclr':
        num02 = len(glob.glob(in_dir + '*-M*C02_*.nc'))
        num03 = len(glob.glob(in_dir + '*-M*C03_*.nc'))
        num05 = len(glob.glob(in_dir + '*-M*C05_*.nc'))
        if (num02 != num03) or (num03 != num05) or (num02 != num05):
            print('check database')
            print('C02: %i files' % num02 )
            print('C03: %i files' % num03 )
            print('C05: %i files' % num05 )
            dt_list=[]
        else:
            print('database looks good')
            flist = [os.path.basename(x) for x in glob.glob(in_dir + '*-M*C02_*.nc')]
            dt_list = [y.split('_')[3][1:12] for y in flist]
    elif im_type is 'fclr':
        num01 = len(glob.glob(in_dir + '*-M*C01_*.nc'))
        num03 = len(glob.glob(in_dir + '*-M*C03_*.nc'))
        num05 = len(glob.glob(in_dir + '*-M*C05_*.nc'))
        if (num01 != num03) or (num03 != num05) or (num01 != num05):
            print('check database')
            print('C01: %i files' % num01 )
            print('C03: %i files' % num03 )
            print('C05: %i files' % num05 )
            dt_list=[]
        else:
            print('database looks good')
            flist = [os.path.basename(x) for x in glob.glob(in_dir + '*-M*C01_*.nc')]
            dt_list = [y.split('_')[3][1:12] for y in flist]
    elif im_type is 'dust':
        num11 = len(glob.glob(in_dir + '*-M*C11_*.nc'))
        num13 = len(glob.glob(in_dir + '*-M*C13_*.nc'))
        num15 = len(glob.glob(in_dir + '*-M*C15_*.nc'))
        if (num11 != num13) or (num13 != num15) or (num11 != num15):
            print('check database')
            print('C11: %i files' % num11 )
            print('C13: %i files' % num13 )
            print('C15: %i files' % num15 )
            dt_list=[]
        else:
            print('database looks good')
            flist = [os.path.basename(x) for x in glob.glob(in_dir + '*-M*C11_*.nc')]
            dt_list = [y.split('_')[3][1:12] for y in flist]
    else:
       dt_list=[] 
       raise ValueError("image type is not recognized") 
            
    return dt_list
     


            
def gen_area_def(ad_code=1):
    """
    generate area definition needed for succeeding processes
    https://pyresample.readthedocs.io/en/latest/geo_def.html
    """
    
    
    if ad_code == 1:
        area_id = 'iid_ss_rgb'
        description = 'IID Salton Sea RGB'
        proj_id = 'longlat'
        x_size = 257
        y_size = 119
        area_extent = (-117.5128849,32.49153322,-114.6392916,33.84804877)    # x_LL; y_LL; x_UR; y_UR
        proj_dict = {'proj': 'longlat','ellps':'WGS84', 'units': 'deg', 'datum': 'WGS84', 'no_defs':''}
        area_def = geometry.AreaDefinition(area_id, description, proj_id, proj_dict, x_size, y_size, area_extent)
    elif ad_code == 2:
        area_id = 'iid_ss_dust'
        description = 'IID Salton Sea DUST'
        proj_id = 'longlat'
        x_size = 131
        y_size = 60
        area_extent = (-117.5452031,32.48852851,-114.6155952,33.85622255) 
        proj_dict = {'proj': 'longlat','ellps':'WGS84', 'units': 'deg', 'datum': 'WGS84', 'no_defs':''}
        area_def = geometry.AreaDefinition(area_id, description, proj_id, proj_dict, x_size, y_size, area_extent)
    #elif ad_code == 3:
    #    area_id = 'iid_ss_dust utm'
    #    description = 'IID Salton Sea DUST 26911'
    #    proj_id = 'longlat'
    #    x_size = 132
    #    y_size = 61
    #    area_extent = (724091.06,3594555.376,448778.264,3748795.706) 
    #    proj_dict = {'a': 6378137, 'proj': 'utm', 'zone': '11', 'ellps':'GRS80', 'datum': 'NAD83', 'units': 'm', 'no_defs':''}
    #    area_def = geometry.AreaDefinition(area_id, description, proj_id, proj_dict, x_size, y_size, area_extent)
    #elif ad_code == 3:
    #    area_id = 'iid_dust_stere'
    #    description = 'IID Salton Sea DUST coord2area_def'
    #    proj_id = 'stere'
    #    x_size = 110
    #    y_size = 61
    #    area_extent = (-137675.82522482562, -74889.46371911185, 137675.82522482562, 76802.66327246893)
    #    proj_dict = {'proj':'stere', 'lat_0':33.17237553, 'lon_0':-116.08039915, 'ellps':'WGS84'}
    #    area_def = geometry.AreaDefinition(area_id, description, proj_id, proj_dict, x_size, y_size, area_extent)
    else:
        raise ValueError("invalid area definition")  
    
    return area_def
        
    
    

def gen_comps(in_dir, out_dir, dt_list, im_type='rgb', ad_code=1):
    """
    simple wrapper to generate composites from 
    either gen_rgb or gen_dust
        
    Parameters:
    ----
    in_dir:  str
        the folder where GOES-R images are
    out_dir: str
        the folder where output geotiff goes
    dt_list: list
        the date and time combination used to generate the composite
    im_type: str
        either 'rgb' or 'dust'
    """
    
    
    area_def = gen_area_def(ad_code)
    for dt in dt_list:
        in_files = glob.glob(in_dir + '*' + dt + '*.nc')
        out_file = out_dir + dt + '.tif'
        
        if im_type is 'rgb':
            gen_rgb(in_files, out_file, area_def)
        elif im_type is 'dust':
            gen_dust(in_files, out_file, area_def)
        elif im_type is 'nclr':
            gen_nclr(in_files, out_file, area_def)




def gen_rgb(in_files, out_file, area_def):
    """
    generate "true-color" composite, and save as goetiff
        
    Parameters:
    ----
    in_files:  list
        list of GOES-R images to generate "true color" composite
    out_file: str
        the output geotiff file
    """
    
    
    scn = Scene(reader='abi_l1b', filenames = in_files)
    scn.load(['true_color'])
    new_scn = scn.resample(area_def, resampler='nearest')
    new_scn.save_dataset('true_color', filename=out_file, writer='geotiff') #, dtype='uint8')
    
    print('generated' + out_file)
    print()
        



def gen_nclr(in_files, out_file, area_def):
    """
    generate "natural-color" composite, and save as goetiff
        
    Parameters:
    ----
    in_files:  list
        list of GOES-R images to generate "natural color" composite
    out_file: str
        the output geotiff file
    """
    
        
    scn = Scene(reader='abi_l1b', filenames = in_files)
    scn.load(['natural_color'])
    new_scn = scn.resample(area_def, resampler='nearest')
    new_scn.save_dataset('natural_color', filename=out_file, writer='geotiff') #, dtype='uint8')
    
    print('generated' + out_file)
    print()
    


    
def gen_dust(in_files, out_file, area_def):
    """
    generate "dust" composite, and save as goetiff
        
    Parameters:
    ----
    in_dir:  str
        the folder where GOES-R images are
    out_dir: str
        the folder where output geotiff goes
    dt_list: list
        the date and time combination used to generate the composite
    """
    
    
    scn = Scene(reader='abi_l1b', filenames = in_files)
    scn.load(["dust"])
    #new_scn = scn.resample('northamerica', resampler='nearest')
    new_scn = scn.resample(area_def, resampler='nearest')
    new_scn.save_dataset("dust", filename=out_file, writer='geotiff')#, dtype='uint8')
    
    print('generated' + out_file)
    print()




def gen_bands(in_dir, out_dir, dt_list, im_type='fclr', ad_code=1):
    """
    another wrapper to generate bands from gen_band
    this allows us to do other processing instead the default ones from satpy
        
    Parameters:
    ----
    in_dir:  str
        the folder where GOES-R images are
    out_dir: str
        the folder where output geotiff goes
    dt_list: list
        the date and time combination used to generate the composite
    im_type: str
        'fclr'
    """
    
    
    area_def = gen_area_def(ad_code)
    for dt in dt_list:
        in_files = glob.glob(in_dir + '*' + dt + '*.nc')
        gen_band(in_files, out_dir, dt, im_type, area_def)
        



def gen_band(in_files, out_dir, dt, im_type, area_def):
    """
    generate radiance of a single band, and save as goetiff
        
    Parameters:
    ----
    in_files:  list
        GOES-R L1b radiance for a specific date-time 
    out_dir: str
        where the output geotiff files go
    dt: str
        date-time combo
    im_type: str
        'fclr'
    """
    
          
    scn = Scene(reader='abi_l1b', filenames = in_files)
    
    if im_type is 'fclr':
        bnds = ['C01', 'C03', 'C05']
    elif im_type is 'dust':
        bnds = ['C11', 'C13', 'C15']
    else:
        raise ValueError("incorrect image type")
        
    for bnd in bnds:
        scn.load([bnd])
        new_scn = scn.resample(area_def, resampler='nearest')
        out_file = out_dir + dt + '_' + bnd + '.tif'
        if im_type is 'dust':
            new_scn.save_dataset(bnd, filename=out_file, writer='geotiff', enhance=False, dtype=np.float32)
            print('generated' + out_file)
            print()
        else:
            new_scn.save_dataset(bnd, filename=out_file, writer='geotiff')#, enhance=False, dtype=np.float32)
            print('generated' + out_file)
            print()

        






    #print('renamed total of %i files' % r)
    #print('skipped total of %i files' % s)   