# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:31:44 2017

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


def vi_calc(img_filepath, output = True, out_filepath=None,
            b1_pos=3, b2_pos=0, mask_val=-1):
    """
    simple normalized vegetation index calculator
    """
    
    ext = os.path.splitext(img_filepath)[-1]
    if out_filepath is None:
        if ext == ".tif":
            out_filepath = img_filepath.replace('.tif', '_ndvi.tif')
        else:
            print ("ERROR: check input and output filenames")
            return        
        
    img = improc.imops.imio.imread(img_filepath)
    b1 = img[:, :, b1_pos].astype("float32")
    b2 = img[:, :, b2_pos].astype("float32")
    
    b1y2 = b1 + b2    
    b1 = np.ma.masked_where(b1y2<=0, b1)
    vi = (b1 - b2) / (b1 + b2)
    vi[vi.mask] = mask_val
    vi = vi.astype("float32")
    
    if output is True:
        improc.gis.rastertools.write_geotiff_with_source(img_filepath, vi, out_filepath)

    return vi    
    
    
def classi_loc_max(img_filepath, bg_thres, out_filepath=None, bg_value=-1.,
                   img_enh=False, use_otsu=False, use_adapt=True,                    
                   min_distance=2, output_lab=False, output_bw=False,
                   rmv_sm_holes=False, output_cov=True):
    """
    adaptive threshold & watershed segmentation based procedure
    
    Parameters
    ----------
    img_filepath: str
        fullpath of the input imagery
    bg_thres: float
        background value to be ignored in the process
    out_filepath: str
        fullpath of the output vegetation extraction file
    bg_value: float/int
        value used to fill all the ignored pixels
    img_enh: bool
        use image contract enhancement or not (default to False)
        currently, the equlization adaptive histogrm is used as an option
        when Otsu threshold is chosen
    use_otsu: bool
        use Otsu alrorithm to decide threshold (default to False)
    use_adapt: bool
        use adaptive threshold (default to True)
    min_distance: int
        min_distance parameter used in finding local maximum
    output_lab: bool
        output label (from watershe segmentation) as a file or not
    output_bw: bool
        output bw (from thresholding) as a file or not
    rmv_sm_holes: bool
        remove small holes in the results or not
    output_cov: bool
        output results as a binary (0/1 ; soil/veg) file or not    
    """
    
    start_time = time.time()
    print('processing: ' + img_filepath)
    
    # deal with input data first
    if 'ndvi' in img_filepath:
        img = improc.imops.imio.imread(img_filepath)
    else:
        rgb = improc.imops.imio.imread(img_filepath)
        if rgb.shape[2] != 4:
            print('check rgb image file')
            return
        img = vi_calc(img_filepath, output=False)
        rgb_sum = np.sum(rgb, axis=2)    
    
    # setting up output filenames structure   
    if out_filepath is None:    #use default output filename
        out_filepath = os.path.splitext(img_filepath)[0] + '_loc_cov.tif'
    
    if output_lab:
        if ('cov' in out_filepath):
            lab_filepath = out_filepath.replace('cov','label')
        else:
            print('check filenames')
            return
        
    if rmv_sm_holes:
        if ('cov' in out_filepath):
            rmv_filepath = out_filepath.replace('cov','rmv')
        else:
            print('check filenames')
            return
       
    
    #img = img_read(img_filepath)
    #img = np.ma.masked_less_equal(img, bg_thres)
    #img[img.mask] = bg_value
    
    #anisodiff2y3d transformation / enhancement
    #img = anisodiff2y3d.anisodiff(img, niter=3, kappa=80, gamma=0.2)
    
    #gaussian filter    
    #img = skimage.filters.gaussian(img, sigma=0.5)     
    
    #image transformation from float to 8bit int (0-255)
    #img = (img-np.min(img)) * 255 / (np.max(img) - np.min(img))
    #img = img.astype(int)
    
    if use_otsu:
        if img_enh:
            img = skimage.exposure.equalize_adapthist(img)  #skimage contrast enhancement
            #write_geotiff_w_source(img_filepath, img, img_filepath.replace('.tif','_enh.tif'))
        #image transformation from float to 8bit int (0-255)
        img = (img-np.min(img)) * 255 / (np.max(img) - np.min(img))
        img = img.astype(int)
        val = skimage.filters.threshold_otsu(img)
        print("Otsu value: ", val)
        bw = img > val
    elif use_adapt:   #use adaptive threshold
        img = np.ma.masked_less_equal(img, bg_thres)
        img[img.mask] = bg_value       
        bw = skimage.filters.threshold_adaptive(img, 11)
    else:   #use global threshold
        bw = img > bg_thres
        
    #remove really bright pixels, likely bright sand
    #but requires rgb imagery to do it
    if not ('ndvi' in img_filepath):    
        #refl_roof = np.average(rgb_sum) - 0.1*np.std(rgb_sum)
        refl_roof = np.max(rgb_sum) - 1.25*np.std(rgb_sum)
        rgb_sum = np.ma.masked_greater(rgb_sum, refl_roof)
        bw[rgb_sum.mask] = False
    
    distance = scipy.ndimage.distance_transform_edt(bw)
    loc_max = skimage.feature.peak_local_max(distance, indices=False, 
                                             min_distance=min_distance,
                                             labels=bw)
    markers = scipy.ndimage.label(loc_max)[0]
    labels = skimage.morphology.watershed(-distance, markers, mask=bw)
    
    #results output    
    if output_bw:
        bw = bw.astype('float32')
        write_geotiff_w_source(img_filepath, bw, img_filepath.replace('.tif','_bw.tif'))
        
    if output_lab:
        write_geotiff_w_source(img_filepath, labels, lab_filepath)
        
    if output_cov:
        labels[labels > 0] = 1.
        labels[labels <= 0] = 0.        
        labels = labels.astype('float32')
    
    write_geotiff_w_source(img_filepath, labels, out_filepath)    

    if rmv_sm_holes:
        labels_bool = labels.astype('bool')
        cov_rmv_sm = skimage.morphology.remove_small_holes(labels_bool)
        cov_rmv_sm = cov_rmv_sm.astype('float32')
        write_geotiff_w_source(img_filepath, cov_rmv_sm, rmv_filepath)   
            
    #create processing log
    log_filepath = out_filepath.replace(os.path.splitext(out_filepath)[-1], '.log')
    log = open(log_filepath, 'w')
    log.write(" input_file:%s\n" %(img_filepath))
    log.write(" background_threshold=%s\n" %(bg_thres))
    log.write(" use adapt:%s\n" %(use_adapt))    
    log.write(" use Otsu:%s\n" %(use_otsu))
    log.write(" img_enhance:%s\n" %(img_enh))    
    log.write(" min_distance=%s\n" %(min_distance))
    log.write(" remove_small_holes:%s\n" %(rmv_sm_holes))
    log.write(" %s seconds" %(time.time() - start_time))    
    log.close()
    
    print("generated: " + out_filepath)    
    print("in --- %.2f seconds ---" % (time.time() - start_time))
    
    
def distance_map(img_filepath, bg_thres, cov_filepath, seg_filepath=None,
                 bg_value=-1, radius=1, use_adaptive=True, use_gaussian=False):
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