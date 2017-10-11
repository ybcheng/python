# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:01:06 2017

@author: ybcheng

from:
http://forum.developer.parrot.com/t/details-of-irradiance-list-tag-for-sunshine-sensor-in-exif-data-of-sequoia/5261/46
"""


import sys
import os
import glob
import exiftool
import base64
import struct
import shutil
import copy


import improc

import pandas as pd
import numpy as np


def proc_ms_decipher (in_dir, metadata, out_dir=None, int_time_scale=1000000):
    """
    process parrot sequoia imagery for testing
    """
    
    meta_df = pd.read_csv(metadata, sep=',')
    
    for i in np.arange(meta_df.shape[0]):
        in_file = in_dir + meta_df.Filename[i]
        
        # pre-calculated formula: DN = a + b * int_time        
        if "GRE" in meta_df.Filename[i]:
            a = 0.   #BB2; #BB1: 41588
            b = 0.   #BB2; #BB1: 134.11
        elif "RED" in meta_df.Filename[i]: 
            a = 43342  #BB3_post; BB3_pre: 24514; BB2: 37074; #BB1: 21670
            b = 293.62   #BB3_post; BB3_pre: 182.66; BB2: 296.49; #BB1: 149.25
        elif "REG" in meta_df.Filename[i]:
            a = 0.   #BB2; #BB1: 8201.5            
            b = 0.   #BB2; #BB1: 30.488
        elif "NIR" in meta_df.Filename[i]:
            a = 25680  #BB3_post; BB3_pre: 15025; BB2: 21572; #BB1: 14137
            b = 96.797  #BB3_post; BB3_pre: 89.943; BB2: 157.01; #BB1: 73.487
        else:
            print ("ERRROR:  " + meta_df.Filename[i])            

        im = improc.imops.imio.imread(in_file)
        
        int_time_nu = meta_df.Shutterspeed[i].split("/")[0]
        int_time_de = meta_df.Shutterspeed[i].split("/")[1]
        int_time = float(int_time_nu) / float(int_time_de)
        int_time = int_time * int_time_scale

        if (a == 0. and b == 0.):
            refl_coeff = 0.
        else:
            refl_coeff = 0.6 / (a + int_time * b)
        
        out_img = im * refl_coeff
        
        if out_dir is None:
            out_dir = in_dir + "refl/"
        
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            
        out_file = out_dir + meta_df.Filename[i].replace('TIF', 'tif')
        improc.imops.imio.imsave(out_file, out_img)
        
        print ('processed:  ' + out_file)


def stack_ms(basedir):
    """
    designed to stack Parrot Sequoia MS imagery
    
    Parameters
    ----------
    basedir: str
        directory where the bands are stored
    """
    
    grn_files = glob.glob(basedir + '*GRE*tif')
    red_files = glob.glob(basedir + '*RED*tif')
    reg_files = glob.glob(basedir + '*REG*tif')
    nir_files = glob.glob(basedir + '*NIR*tif')
    
    for (g,r,e,n) in zip(grn_files, red_files, reg_files, nir_files):
        temp = improc.imops.imio.imread(g)
        stacked = np.empty(temp.shape + (4,), temp.dtype)
        stacked[:,:,0] = copy.deepcopy(temp)
        
        temp = improc.imops.imio.imread(r)
        stacked[:,:,1] = copy.deepcopy(temp)
        temp = improc.imops.imio.imread(e)
        stacked[:,:,2] = copy.deepcopy(temp)
        temp = improc.imops.imio.imread(n)
        stacked[:,:,3] = copy.deepcopy(temp)
               
        improc.imops.imio.imsave(g.replace('GRE.tif','stck.tif'), stacked)


def gen_refl_imu (metadata, rgb_imu, refl_imu, replace=True):
    """
    carefule using this one
    very unflexible
    """
    
    meta_df = pd.read_csv(metadata, sep=',')
    rgb_df = pd.read_csv(rgb_imu, sep=',', header=None)
    
    refl_df = pd.DataFrame(columns=['filename','lat','lon','elev'])
    refl_df.filename = meta_df.Filename
    
    for i in np.arange(rgb_df.shape[0]):
        rgb_name = rgb_df.iloc[i,0]
        f_pattern = rgb_name[17:23]        
        #f_pattern = rgb_name.replace('_RGB.JPG','_')
        
        for j in np.arange(refl_df.shape[0]):
            refl_name = refl_df.filename[j]
            if f_pattern in refl_name:
                refl_df.lat[j] = rgb_df.iloc[i,16]
                refl_df.lon[j] = rgb_df.iloc[i,17]
                refl_df.elev[j] = rgb_df.iloc[i,18]
                
    if (os.path.exists(refl_imu) and replace):
        os.remove(refl_imu)
        
    refl_df.to_csv(refl_imu, index=False, sep=',')
    



            


"""
followings are some quick command to copy and rename UAV imagery for futhre processing
"""
"""
files = glob.glob('M:/UAV_imagery/20170610_OwensLake_Ph7aMV/T36_1/T36-1w_5/*.JPG')
len(files)
for f in files:
    o = f.replace('T36-1w_5\\', 'flight5_')
    shutil.copy2(f,o)    












uint64_t timestamp (us)
uint16_t CH0 (count)
uint16_t CH1 (count)
uint16_t gain index
uint16_t integration time (ms)
float    yaw
float    pitch
float    roll


irradiance_list_tag = 'XMP:IrradianceList'
irradiance_calibration_measurement_golden_tag = 'XMP:IrradianceCalibrationMeasurementGolden'
irradiance_calibration_measurement_tag = 'XMP:IrradianceCalibrationMeasurement'

tags = [ irradiance_list_tag, irradiance_calibration_measurement_tag ]

directory = 'test'

channels = [ 'RED', 'NIR' ]

index = 0

for channel in channels:
    files = glob.glob(os.path.join(directory, '*' + channel + '*'))
    with exiftool.ExifTool() as et:
        metadata = et.get_tags_batch(tags, files)
        for file_metadata in metadata:
            irradiance_list = file_metadata[irradiance_list_tag]
            irradiance_calibration_measurement = file_metadata[irradiance_calibration_measurement_tag]

            irradiance_list_bytes = base64.b64decode(irradiance_list)

            print(files[index])
            index += 1

            for irradiance_data in struct.iter_unpack("qHHHHfff", irradiance_list_bytes):
                print(irradiance_data)
"""