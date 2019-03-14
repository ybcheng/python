# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:49:10 2017

@author: ybcheng
"""

#import glob
import os
#import sys
#import pandas as pd
import numpy as np
import scipy
#import rastertools
import rasterio
import time
#import copy
#import fiona
#import shapely
import skimage
import cv2
#import functools
#import shutil
#import matplotlib.pyplot as plt
import improc
import anisodiff2y3d

from scipy import stats
from skimage import filters, morphology, feature

#from ..imops import imio, imcalcs
#from ..gis import rastertools, shapetools, extract
#from ..gen import strops, dirfuncs, wrappers
#from ..cv import classify, slicing
#from ..dbops import finder, loader, parse
#from . import anisodiff2y3d
#from improc.cv.genutils import square, disk


def img_read(filename):
    """
    generic function to read in image files and 
    store as numpy array
    
    NOTE: currently ONLY support tif format
    """
    
    ext = os.path.splitext(filename)[-1] 
    if ext == ".tif" or ext == ".tiff" or ext == ".TIF":
        #try:
        image = rasterio.open(filename).read()
        
        #this part is to make sure im in [X, Y, Z] order            
        if len(image.shape) == 3:
            shuffled_image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
            if shuffled_image.shape[2] == 1:
                shuffled_image = shuffled_image[:, :, 0]
    else:
        print("File format is not currently supported.")
        return

    return shuffled_image
    

def write_geotiff_w_source(source_filename, output_img, output_filename,
                              nodata=0, compress=True, extra_tags=None):
    """
    Uses the info in the input file (source, in geotiff format) to
    write a new geotiff (output) with the results after some processing
    that's been done to the input file, but otherwise the same geo information.
    
    Note: surpports GeoTIFF format as indicated
    """

    try:
        source = rasterio.open(source_filename, "r")
    except IOError:
        print("Error opening file %s. Returning." % source_filename)
        return
    
    if output_img.ndim == 2:
        rev_output_img = np.ma.resize(output_img, output_img.shape + (1,))
    else:
        rev_output_img = output_img    
    
    if compress:
        compress = "lzw"
    else:
        compress = None

    out_raster = rasterio.open(output_filename, "w", driver="GTiff",
            width=rev_output_img.shape[1], height=rev_output_img.shape[0],
            count=rev_output_img.shape[2], dtype=rev_output_img.dtype,
            transform=source.affine, crs=source.crs, nodata=nodata,
            compress=compress)

    for i in range(rev_output_img.shape[2]):
        out_raster.write_band(i + 1, rev_output_img[:, :, i])

    if extra_tags is not None:
        out_raster.update_tags(**extra_tags)

    source.close()
    out_raster.close()



def testing_classi_loc_max(img_filepath, bg_thres, out_filepath=None, min_distance=3,
                      use_otsu=True, use_adapt=False, output_cov=True):
    """
    adaptive threshold & watershed segmentation based procedure
    """
    
    start_time = time.time()
    print('processing: ' + img_filepath)

    if out_filepath is None:
        out_filepath = os.path.splitext(img_filepath)[0] + '_loc_cov.tif'

    img = improc.imops.imio.imread(img_filepath)
    img = img_read(img_filepath)
   
    #img = np.ma.masked_less_equal(img, bg_thres)
    #img[img.mask] = 0.0
    
    if use_adapt:
        bw = skimage.filters.threshold_adaptive(img, 1111)
    elif use_otsu:
        otsu_val = skimage.filters.threshold_otsu(img)
        bw = img > otsu_val
    else:
        bw = img > bg_thres            
    
    distance = scipy.ndimage.distance_transform_edt(bw)
    loc_max = skimage.feature.peak_local_max(distance, indices=False, 
                                             min_distance=min_distance,
                                             labels=bw)
    markers = scipy.ndimage.label(loc_max)[0]
    labels = skimage.morphology.watershed(-distance, markers, mask=bw)
        
    if output_cov:
        labels[labels > 0] = 1.
        labels[labels <= 0] = 0.
        labels = labels.astype('float32')
    
    write_geotiff_w_source(img_filepath, labels, out_filepath)
    
    print("generated: " + out_filepath)    
    print("in --- %.2f seconds ---" % (time.time() - start_time))
    
    
def classi_loc_max(img_filepath, bg_thres, out_filepath, min_distance=3,
                   output_cov=True):
    """
    adaptive threshold & watershed segmentation based procedure
    """
        
    #img = improc.imops.imio.imread(img_filepath)
    img = img_read(img_filepath)
    img = np.ma.masked_less_equal(img, bg_thres)
    img[img.mask] = 0.0
    
    bw = skimage.filters.threshold_adaptive(img, 11)
    distance = scipy.ndimage.distance_transform_edt(bw)
    loc_max = skimage.feature.peak_local_max(distance, indices=False, 
                                             min_distance=min_distance,
                                             labels=bw)
    markers = scipy.ndimage.label(loc_max)[0]
    labels = skimage.morphology.watershed(-distance, markers, mask=bw)
        
    if output_cov:
        labels[labels > 0] = 1.
        labels[labels <= 0] = 0.
        labels = labels.astype('float32')
    
    write_geotiff_w_source(img_filepath, labels, out_filepath)
    
    
def testing_distance_map(img_filepath, bg_thres, cov_filepath=None, seg_filepath=None,
                 bg_value=-1, radius=1, use_adapt=False, use_otsu=True,
                 use_gaussian=False):
    """
    distance map based procedure
    """
    
    start_time = time.time()
    print('processing: ' + img_filepath)
        
    if not os.path.exists(img_filepath):
        print("ERROR: check input file path")
        return
    
    if cov_filepath is None:
        cov_filepath = os.path.splitext(img_filepath)[0] + '_dis_cov.tif'
    
    img = improc.imops.imio.imread(img_filepath)
    
    #img = img_read(img_filepath)    
    #bg_thres = 0.38
    #bg_value = -1
    
    if use_gaussian:
        img = filters.gaussian_filter(img, sigma=3)    
    
    #img = np.ma.masked_less_equal(img, bg_thres)    
    
    if use_adapt:
        bw = skimage.filters.threshold_adaptive(img, 1111)        
    elif use_otsu:
        otsu_val = skimage.filters.threshold_otsu(img)
        print(otsu_val)
        bw = img > otsu_val
    else:
        bw = img > bg_thres 
     
    img[~bw] = bg_value
        
    #radius = 1
    #width = 2 * radius + 1
    #exp_sq = np.ones((width, width), dtype=np.unit8)
    #exp_sq = improc.cv.genutils.square(2 * radius + 1)
    exp_dsk = skimage.morphology.disk(radius)
    #exp_dsk = skimage.morphology.ball(radius)[:,:,0] 
    
    #seg_img = scipy.ndimage.grey_dilation(ndsi_img_mskd, footprint=exp_sq)
    seg_img = scipy.ndimage.grey_dilation(img, footprint=exp_dsk)
    cov_img = np.empty(seg_img.shape, 'uint8')
    seg_img = np.ma.masked_equal(seg_img, -1.0)
    cov_img[seg_img.mask] = 0
    cov_img[~seg_img.mask] = 1
    
    if seg_filepath is not None:
        write_geotiff_w_source(img_filepath, seg_img, seg_filepath)
    write_geotiff_w_source(img_filepath, cov_img, cov_filepath)
    
    #log_filepath = cov_filepath.replace(os.path.splitext(cov_filepath)[1], '.log')
    #log = open(log_filepath, 'w')
    #log.write(" input_file:%s\n" %(img_filepath))
    #log.write(" background_threshold=%s\n" %(bg_thres))
    #log.write(" radius=%s\n" %(radius))
    #log.write(" use_adaptive:%s\n" %(use_adaptive))
    #log.write(" %s seconds" %(time.time() - start_time))    
    #log.close()
    
    print("generated: " + cov_filepath)
    print("--- %.2f seconds ---" % (time.time() - start_time))
    
    
def distance_map(img_filepath, bg_thres, cov_filepath, seg_filepath=None,
                 bg_value=-1, radius=1, use_adaptive=False, use_gaussian=False):
    """
    distance map based procedure
    """
    
    start_time = time.time()
    
    img = img_read(img_filepath)    
    #bg_thres = 0.38
    #bg_value = -1
    
    if use_gaussian:
        img = filters.gaussian_filter(img, sigma=3)    
    
    img = np.ma.masked_less_equal(img, bg_thres)    
    
    if use_adaptive:
        bw = skimage.filters.threshold_adaptive(img, 11)
        bw[img.mask] = False
        img[~bw] = bg_value
    else:
        img[img.mask] = bg_value
     
    #radius = 1
    #width = 2 * radius + 1
    #exp_sq = np.ones((width, width), dtype=np.unit8)
    #exp_sq = improc.cv.genutils.square(2 * radius + 1)
    exp_dsk = skimage.morphology.disk(radius)
    #exp_dsk = skimage.morphology.ball(radius)[:,:,0] 
    
    #seg_img = scipy.ndimage.grey_dilation(ndsi_img_mskd, footprint=exp_sq)
    seg_img = scipy.ndimage.grey_dilation(img, footprint=exp_dsk)
    cov_img = np.empty(seg_img.shape, 'uint8')
    seg_img = np.ma.masked_equal(seg_img, -1.0)
    cov_img[seg_img.mask] = 0
    cov_img[~seg_img.mask] = 1
    
    if seg_filepath is not None:
        write_geotiff_w_source(img_filepath, seg_img, seg_filepath)
    write_geotiff_w_source(img_filepath, cov_img, cov_filepath)
    
    log_filepath = cov_filepath.replace(os.path.splitext(cov_filepath)[1], '.log')
    log = open(log_filepath, 'w')
    log.write(" input_file:%s\n" %(img_filepath))
    log.write(" background_threshold=%s\n" %(bg_thres))
    log.write(" radius=%s\n" %(radius))
    log.write(" use_adaptive:%s\n" %(use_adaptive))
    log.write(" %s seconds" %(time.time() - start_time))    
    log.close()
    
    print("--- %.2f seconds ---" % (time.time() - start_time))