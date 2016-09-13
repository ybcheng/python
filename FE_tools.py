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
import skimage
import cv2
import functools
import shutil
import matplotlib.pyplot as plt
from scipy import stats
from skimage import filters, morphology, feature

from ..imops import imio, imcalcs
from ..gis import rastertools, shapetools, extract
from ..gen import strops, dirfuncs, wrappers
from ..cv import classify, slicing
from ..dbops import finder, loader, parse
from . import anisodiff2y3d


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