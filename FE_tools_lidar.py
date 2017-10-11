# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:23:05 2017

@author: ybcheng
"""



import sys
import os
import glob
import exiftool
import base64
import struct
import shutil
import copy
import math
import time
import gdal

import improc

import pandas as pd
import numpy as np


def proc_lascanopy (in_file, grid_size, cut_off, out_file):
    """
    lascanopy like tool
    """
        
    start_time = time.time()
    
    in_df = pd.read_csv(in_file, sep=None, header=None, engine='python')
    in_df.columns = ['x' , 'y', 'z']
    
    in_df['veg'] = in_df.iloc[:,2].apply(lambda x: x>cut_off)
    in_df['veg'] = in_df['veg'].astype(int)
    
    init_x = math.ceil(np.min(in_df.x))
    init_y = math.ceil(np.min(in_df.y))
    
    dim_x = math.floor((np.max(in_df.x)-init_x)/grid_size) + 1
    dim_y = math.floor((np.max(in_df.y)-init_y)/grid_size) + 1
    
    end_x = init_x + (dim_x - 1) * grid_size
    end_y = init_y + (dim_y - 1) * grid_size
    
    out_df = pd.DataFrame(index=np.arange(dim_x*dim_y),
                          columns=['x', 'y','cov'], dtype=float)
                          
    out_df.x = list(np.arange(init_x, end_x+grid_size, grid_size)) * dim_y
    out_df.y = sorted(list(np.arange(init_y, end_y+grid_size, grid_size)) * dim_x)
   
    for i in np.arange(out_df.shape[0]):
        if ((i%1000) == 0):
            print("%i / %i --- %.2f seconds" % (i, out_df.shape[0], (time.time()-start_time)))
                    
        up_x = out_df.x[i] + 0.5*grid_size
        lo_x = out_df.x[i] - 0.5*grid_size
        up_y = out_df.y[i] + 0.5*grid_size
        lo_y = out_df.y[i] - 0.5*grid_size
        
        tmp_df = in_df.loc[(in_df.x > lo_x) & (in_df.x < up_x) &
                           (in_df.y > lo_y) & (in_df.y < up_y)]
        
        if (len(tmp_df) == 0):
            out_df.loc[i,'cov'] = -1
        else:
            out_df.loc[i, 'cov'] = np.average(tmp_df.veg)                   
    
    out_df.to_csv(out_file, index=False, sep=',')
    
    print("Done in --- %.2f seconds ---" % (time.time() - start_time))
    
    return out_file
    
    
def csv2tif(source, target):
    """
    Could be handle little tool to convert csv results from proc_lascanopy
    to GeoTIFF
    
    https://gis.stackexchange.com/questions/177061/ascii-file-with-latitude-longitude-and-data-to-geotiff-using-python
    """
    
    
    cvs = gdal.Open(source)
    if cvs is None:
        print('ERROR: Unable to open %s' % source)
        return

    geotiff = gdal.GetDriverByName("GTiff")
    if geotiff is None:
        print('ERROR: GeoTIFF driver not available.')
        return

    options = []
    geotiff.CreateCopy(target, cvs, 0, options)
