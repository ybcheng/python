# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 12:22:37 2016

@author: ybcheng
"""


import glob
import os
import sys
import pandas as pd
import numpy as np
import scipy
import time
import copy
import fiona
import shapely
import rasterio
import skimage
import cv2
import functools
import shutil
import matplotlib.pyplot as plt
from scipy import stats
from skimage import filters, morphology, feature

from improc.imops import imio, imcalcs
from improc.gis import rastertools, shapetools, extract, mask
from improc.gen import strops, dirfuncs, wrappers
from improc.cv import classify, slicing
from improc.dbops import finder, loader, parse
#from . import anisodiff2y3d


def stack_OLI(basedir):
    """
    designed to stack OLI B2 thru B6 together
    
    Parameters
    ----------
    basedir: str
        directory where the bands are stored
    """
    
    blu_files = glob.glob(basedir + '*B2*TIF')
    grn_files = glob.glob(basedir + '*B3*TIF')
    red_files = glob.glob(basedir + '*B4*TIF')
    nir_files = glob.glob(basedir + '*B5*TIF')
    sw1_files = glob.glob(basedir + '*B6*TIF')
    
    for (b,g,r,n,s) in zip(blu_files, grn_files, red_files, nir_files, sw1_files):
        temp = imio.imread(b)
        stacked = np.empty(temp.shape + (5,), temp.dtype)
        stacked[:,:,0] = copy.deepcopy(temp)
        
        temp = imio.imread(g)
        stacked[:,:,1] = copy.deepcopy(temp)
        temp = imio.imread(r)
        stacked[:,:,2] = copy.deepcopy(temp)
        temp = imio.imread(n)
        stacked[:,:,3] = copy.deepcopy(temp)
        temp = imio.imread(s)
        stacked[:,:,4] = copy.deepcopy(temp)
       
        rastertools.write_geotiff_with_source(b, stacked, b.replace('.TIF', '_stck.TIF'),
                                              nodata=0, compress=False)
                                              
        print('processed: %s' % b)
        
        
def extract_average(filepaths, shapefile):
    """
    little utility that extracts and reports average raster values
    within a polygon shapefile
    ONLY works for single-band images
    
    Parameters
    ----------
    filepaths: list
        files to be processed
    shapefile: str
        polygon shapefile
    """
    
    for f in filepaths:
        raster = rasterio.open(f)
        polygon = shapetools.join_layers(shapefile)
        shapes = [shapely.geometry.mapping(polygon)]
        masked, geo_affine = mask.mask(raster, shapes, crop=True)
        if masked.shape[0] != 1 and masked.shape[1] != 1 and masked.shape[2] != 1:
            print('check image dimension: %s' % f)
        else:
            print(np.average(masked))