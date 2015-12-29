# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:17:58 2015
@author: Yen-Ben Cheng
simple functions that operates on IDS images:
# prepare aligned IDS images for Photoscan
# take care of NaN values in IDS images
# generates chl images
!!!ATTENTION!!! functions may not be compatible with tetracam or other images
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
import matplotlib.pyplot as plt
from scipy import stats
from skimage import filters, morphology, feature

from ..imops import imio
from ..gis import rastertools, shapetools, extract
from ..gen import strops
from ..cv import classify, slicing
from ..dbops import finder, loader
from . import anisodiff2y3d


#==============================================================================
# this section is mostly utilities for preprocessing and etc

def gen_imu2(file_path, imufile, imufile2, ext='.tif'):
    """
    the aligned IDS images from matlab have different file names
    this function is designed to replace the old file names in 
    the imu file with the new file names
    
    Parameters:
    ----
    filepath: str
        the folder where the aligned files are
    ext: str
        should be '.tif'
    imufile: str
        the original imu file
    imufile2: str
        the new imu file    
    """
        
    files = glob.glob(file_path + '*' + ext)
    bfiles = pd.DataFrame([os.path.basename(file) for file in files])
    log = pd.read_csv(imufile, sep='\t')
    log['imageNames'] = bfiles[0]
    log.to_csv(imufile2, index=False, sep='\t')
    

def rid_fake_y_flip(input_dir, trim_x, trim_y, output_dir, 
                    ext='.tif', flip=True):
    """
    the aligned IDS2 images have a fake 3rd band, which is can cause issues in
    photoscan mosaics. This function is designed to simplely remove the fake
    3rd band.
    a cropping option is included if one wants to get rid of some
    pixels from the edges.
    a channel flipping option is included since having NIR as the 1st band
    seems to improve mosaic quality. So if one does not like the band
    order in the input files (e.g. 1st band RED, 2nd band NIR), one can change
    it (e.g. 1st band NIR, 2nd band RED).
    
    Parameters:
    ----
    input_dir: str
        directory of input files
    ext: str
        should be '.tif'
    trim_x: int
    trim_y: int
        crop ceratin number of pixels off the edges
    output_dir: str
        directory to save output files
    flip: boolean
        flip the channel order or not
    """
    
    
    input_files = glob.glob(input_dir + '*' + ext)
    output_files = [output_dir + os.path.basename(input_file) 
                    for input_file in input_files]
    
    for (input_file, output_file) in zip (input_files, output_files):
        img_file = imio.imread(input_file)
        if flip:
            flip_img = np.empty(img_file.shape, img_file.dtype)
            flip_img[:, :, 0] = img_file[:, :, 1]
            flip_img[:, :, 1] = img_file[:, :, 0]
            imio.imsave(output_file,
                        flip_img[trim_y:-trim_y, trim_x:-trim_x, 0:2])
        else:
            imio.imsave(output_file,
                        img_file[trim_y:-trim_y, trim_x:-trim_x, 0:2])
        
       
def flip(input_dir, output_dir, order, ext='.tif'):
    """
    This function is designed to simply change band order of IDS images to 
    experiment if it'll make the mosaic better.
             
    Parameters:
    ----
    input_dir: str
        directory of input files
    ext: str
        should be '.tif'
    order: 
        where to put old bands in the new file
        e.g. if the first band of the new file is the second band of
        the old file, then order should be [1, x, x,...]
        
    output_dir: str
        directory to save output files
    """
    
    
    input_files = glob.glob(input_dir + '*' + ext)
    output_files = [output_dir + os.path.basename(input_file) 
                    for input_file in input_files]
    
    for (input_file, output_file) in zip (input_files, output_files):
        img_file = imio.imread(input_file)
        flip_img = np.empty(img_file.shape, img_file.dtype)
        for i in range(len(order)):
            flip_img[:, :, i] = copy.deepcopy(img_file[:, :, order[i]])
        
        imio.imsave(output_file, flip_img)
                        
                        
def rid_nan(input_dir, output_dir, ext='.tif'):
    """
    seomtimes images made in ENVI contain NaN because of different reasons
    this function simply replace all the NaN with zero
    
    Parameters:
    ----
    input_dir: str
        directory of input files
    ext: str
        should be '.tif'
    output_dir: str
        directory to save output files
    """
    
    
    input_files = glob.glob(input_dir + '*' + ext)
    output_files = [output_dir + os.path.basename(input_file) 
                    for input_file in input_files]
    
    for (input_file, output_file) in zip (input_files, output_files):
        img_file = imio.imread(input_file)
        new_img = np.nan_to_num(img_file)
        imio.imsave(output_file, new_img)


def replace_nan(input_dir, output_dir, ext='.tif', repl_num=-1):
    """
    seomtimes images made in ENVI contain NaN because of different reasons
    this function simply replace all the NaN with a certain number
    
    Parameters:
    ----
    input_dir: str
        directory of input files
    ext: str
        should be '.tif'
    output_dir: str
        directory to save output files
    repl_num: int
        the number you'd like to replace NaN with
    """
    
    
    input_files = glob.glob(input_dir + '*' + ext)
    output_files = [output_dir + os.path.basename(input_file) 
                    for input_file in input_files]
    
    for (input_file, output_file) in zip (input_files, output_files):
        img_file = imio.imread(input_file)
        nan_mask = np.isnan(img_file)
        img_file[nan_mask] = repl_num
        rastertools.write_geotiff_with_source(input_file,
                                              img_file, output_file)
                                    
                                              
def stack(input_dir, output_dir, in_ext='.png', out_ext='.tif'):
    """
    designed to stack all five bands together
    
    Parameters
    ----------
    input_dir: str
        directory where nir bands are stored
    """
    
    nir_files = glob.glob(input_dir + '*' + in_ext)
    output_files = [output_dir + os.path.basename(nir_file) 
                    for nir_file in nir_files]
    
    for i in range(0, len(output_files)):
        output_files[i] = output_files[i].replace(in_ext, out_ext)
    
    red_files = glob.glob(input_dir.replace('IDS NIR', 'IDS Red')+'*'+in_ext)
    svn_files = glob.glob(input_dir.replace('IDS NIR', 'IDS 700')+'*'+in_ext)
    fiv_files = glob.glob(input_dir.replace('IDS NIR', 'IDS 550')+'*'+in_ext)
    fou_files = glob.glob(input_dir.replace('IDS NIR', 'IDS 480')+'*'+in_ext)
    
    for (nir, out) in zip(nir_files, output_files):
        temp = imio.imread(nir)
        stacked = np.empty(temp.shape + (5,), temp.dtype)
        stacked[:,:,0] = copy.deepcopy(temp)
        for red in red_files:
            if os.path.basename(nir)[0:20] == os.path.basename(red)[0:20]:
                temp = imio.imread(red)
                stacked[:,:,1] = copy.deepcopy(temp)
        for svn in svn_files:
            if os.path.basename(nir)[0:20] == os.path.basename(svn)[0:20]:
                temp = imio.imread(svn)
                stacked[:,:,2] = copy.deepcopy(temp)
        for fiv in fiv_files:
            if os.path.basename(nir)[0:20] == os.path.basename(fiv)[0:20]:
                temp = imio.imread(fiv)
                stacked[:,:,3] = copy.deepcopy(temp)
        for fou in fou_files:
            if os.path.basename(nir)[0:20] == os.path.basename(fou)[0:20]:
                temp = imio.imread(fou)
                stacked[:,:,4] = copy.deepcopy(temp)
        imio.imsave(out,stacked)
       
        
#==============================================================================
# this section is about calibration and atmospheric correction
# including IARR, empirical line, and applying cal coeff and SMARTS output
        
def empline_cal(image_filename, spectra_filename, output_filename,
                mask_val = 0):
    """
    empirical line calibration based on the spectral information in a csv file
    
    Parameters
    ----------
    image_filename: str
        full path Filename of the IDS image file
    spectra_filename: str
        csv file contains the spectra information for the calibration
        !!!ATTENTION!!! the file needs to be prepared in a specific format:
        1st line: dark target of the input image file
        2nd line: bright target of the input image file
        3rd line: dark target of the reference
        4th line: bright target of the reference
        each column needs to match each band of the image
        NO header
    output_filename: str
        full path  of Output filename
    mask_val: int
        value for masked pixels
    """


    spectra = pd.read_csv(spectra_filename, header=None)
    image = imio.imread(image_filename)
    
    spectral_axis = imio.guess_spectral_axis(image) 
    if spectral_axis == 0:
        image = imio.axshuffle(image)
    
    output_image = np.empty(image.shape, image.dtype)
    
    for i in range(0, image.shape[2]):
        x = spectra.iloc[0:2, i]
        y = spectra.iloc[2:4, i]
        results = stats.linregress(x,y)
        
        tmp = image[:, :, i].astype('float32')
        tmp = np.ma.masked_equal(tmp, 0)
        tmp = tmp * results[0] + results[1]
        tmp[tmp.mask] = mask_val
        tmp = tmp.astype('float32')
        output_image[:, :, i] = copy.deepcopy(tmp)
        
    rastertools.write_geotiff_with_source(image_filename, output_image, 
        output_filename, nodata=0, compress=False)
        

def empline_cal_files(input_dir, output_dir, img_ext = '.tif', 
                      spectra_ext = 'csv', mask_val = 0):
    """
    simple wrapper to run empirical line calibration on multiple files
    this function assumes one spectral file for one image file, if there are 
    unequal number of image and spectra files, an error will occur
    
    Parameters
    ----------
    input_dir: str
        directory of the IDS image files
    output_dir: str
        directory for output files
    img_ext: str
        file extension for input files, should be '.tif'
    spectra_ext: str
        file extension for the spectra files, should be '.csv'
    mask_val: int
        value for masked pixels
    """

    input_files = glob.glob(input_dir + '*' + img_ext)
    output_files = [output_dir + os.path.basename(input_file) 
                    for input_file in input_files]
    spectra_files = glob.glob(input_dir + '*' + spectra_ext)
    
    for (i, s, o) in zip(input_files, spectra_files, output_files):
        empline_cal(i, s, o, mask_val = mask_val)
        

def calc_iarr_with_geo(filename, iarr_filename):
    """
    simple function that calculates IARR from an image in DN
    
    Parameters
    ----------
    filename: str
        full path of input image in DN
    iarr_filename: str
        fulll path of output IARR image
    """
    
    image = imio.imread(filename)
    
    spectral_axis = imio.guess_spectral_axis(image) 
    if spectral_axis == 0:
        image = imio.axshuffle(image)
    
    image = np.ma.masked_less_equal(image, 0).astype('float32')
    iarr = np.zeros(image.shape, dtype = image.dtype)
    
    for i in range(image.shape[2]):
        iarr[:,:,i] = image[:,:,i]/np.mean(image[:,:,i])
        
    rastertools.write_geotiff_with_source(filename, iarr, iarr_filename,
                                          nodata=-1, compress=False)

    return iarr
    

def set_params():
    """
    Definitions of camera parameters related to radiometric calibration.
    
    This dictionary can (should?) be updated/expanded later to simplify things,
    change "ids" into system name, e.g. "Pomona2",
    and replace "611" "612" with band number "0" "1"
    """

    ids = {
        "611":
            {"sn": 4102887611,
             "filter": "nir",
             "int_time": 0.6,
             "gain": 3.00036E-06,
             "offset": 0},
        "612":
            {"sn": 4102887612,
             "filter": "red",
             "int_time": 0.8,
             "gain": 3.4203E-06,
             "offset": 0},
        "421":
            {"sn": 4102776421,
             "filter": "red_edge",
             "int_time": 1.15,
             "gain": 3.41967E-06,
             "offset": 0},
        "902":
            {"sn": 4102833902,
             "filter": "green",
             "int_time": 0.95,
             "gain": 2.8315E-06,
             "offset": 0},
        "635":
            {"sn": 4102719635,
             "filter": "blue",
             "int_time": 0.92,
             "gain": 4.25458E-06,
             "offset": 0}
        }
        
    return locals()    
        
def dn_2_refl(dn_filename, refl_filename, rad_filename = None, 
              int_time=[1.0,1.0,1.0,1.0,1.0],
              gain=[1.0,1.0,1.0,1.0,1.0],
              offset=[0.0,0.0,0.0,0.0,0.0],
              irrad=[1.0,1.0,1.0,1.0,1.0]):
    """
    function that transfer dn to radiance and reflectance using calibration 
    coefficients and simulated irradiance
    
    Parameters
    ----------
    dn_filename: str
        full path of input image in DN
    rad_filename: str
        fulll path of output radiance image
    refl_filename: str
        full path of output reflectance image
    int_time: float array
        integration time
    gain: float array
        calibration coefficient
    offset: float array
        calibration coefficient
    irrad: float array
        simulated irradiance 
    """
        
    dn_img = imio.imread(dn_filename)
   
    spectral_axis = imio.guess_spectral_axis(dn_img) 
    if spectral_axis == 0:
        dn_img = imio.axshuffle(dn_img)
    
    dn_img = np.ma.masked_less_equal(dn_img, 0).astype('float32')
    rad_img = np.zeros(dn_img.shape, dtype=dn_img.dtype)
    refl_img = np.zeros(rad_img.shape, dtype=rad_img.dtype)
    
    for i in range(dn_img.shape[2]):
        rad_img[:,:,i] = dn_img[:,:,i]/int_time[i] * gain[i] + offset[i]
        refl_img[:,:,i] = rad_img[:,:,i]/irrad[i] * 3.14159265
        
    rad_img[dn_img.mask] = 0.0
    refl_img[dn_img.mask] = 0.0
    
    rastertools.write_geotiff_with_source(dn_filename, refl_img, refl_filename,
                                          nodata=0, compress=False)
    if rad_filename is not None:
        rastertools.write_geotiff_with_source(dn_filename, rad_img, rad_filename,
                                          nodata=0, compress=False)                                      
    return rad_img, refl_img


def dn_2_refl_files(input_dir, img_ext = 'tif', rad = False,
                    #int_time=[1.0,1.0,1.0,1.0,1.0],
                    #gain=[1.0,1.0,1.0,1.0,1.0],
                    #offset=[0.0,0.0,0.0,0.0,0.0],
                    cam_set=['611','612','421','902','635'],
                    irrad=[0.5668,0.7177,0.6621,0.8321,0.9027]):
    """
    simple wrapper to transfer DN to radiance and reflectance on multiple files
    input_dir should be either "masked", "mosaic", "registered",
    "registered masked", or "registered merged"
    all the files in one directory are supposed to be captured with one system
    hence int_time, gain, offset should be good for all the files
    
    Parameters
    ----------
    input_dir: str
        full path of the directory that has all the input files
    img_ext: str
        extension of input files
    (int_time: float array
        integration time
    gain: float array
        calibration coefficient
    offset: float array
        calibration coefficient)
    cam_set: str array
        last three digits of s/n of cameras used, in the order and band number
    irrad: float array
        simulated irradiance, in the order of band number   
    """
    
    dn_files = glob.glob(input_dir + '*' + img_ext)
    # generates output filenames
    refl_files = [dn_file.replace('registered masked', 'physical')
                  for dn_file in dn_files]
    refl_files = [refl_file.replace('registered merged', 'physical')
                  for refl_file in refl_files]
    refl_files = [refl_file.replace('registered', 'physical')
                  for refl_file in refl_files]
    refl_files = [refl_file.replace('mosaic', 'physical')
                  for refl_file in refl_files]
    refl_files = [refl_file.replace('masked', 'physical')
                  for refl_file in refl_files]              
    
    if rad == True:
        rad_files = [refl_file.replace('IDS', 'RAD')
                 for refl_file in refl_files]
    else:
        rad_files = ['None'
                 for refl_file in refl_files]
    
    parameters = set_params()
    cam_set = np.asarray(cam_set)
    int_time = np.empty(cam_set.shape, dtype='float32')
    gain = np.empty(cam_set.shape, dtype='float32')
    offset = np.empty(cam_set.shape, dtype='float32')
    
    for i in range(len(cam_set)):
        int_time[i] = parameters["ids"][cam_set[i]]["int_time"]
        gain[i] = parameters["ids"][cam_set[i]]["gain"]
        offset[i] = parameters["ids"][cam_set[i]]["offset"]

    #print int_time
    #print gain
    #print offset    
                  
    for (dn, rad, refl) in zip(dn_files, rad_files, refl_files):
        dn_2_refl(dn, refl, rad_filename=rad, int_time=int_time, gain=gain,
                  offset=offset, irrad=irrad)        
                  
                  
#==============================================================================
# this section is about chl index calculation and a little post processing
# making histogram like bar chart for reporting

def gen_chl_files(filenames, in_dir, unit='ids', dummy=None, replace=True):
    """
    simple wrapper for chl_with_geo function to generate
    output filenames, and generate chl images in the output folder.
    NOTE: it's a simplified version of improc.prostprocess.gne_ndvi_files
    
    Parameters
    ----------
    filenames: str
        Filename of the IDS5 image file
    in_dir: str
        directory of the IDS5 images, not full pathname, just registered,
        masked, or registered masked etc
    unit: str
        DN files, unit='ids'
        radiance files, unit='rad'
        reflectanc files, unit='refl'
    """
        
    for filename in filenames:
        if (filename.endswith(".tif") and (unit in filename.lower())):
            # generate a good output filename
            #chl_filename = strops.ireplace("IDS", "chl", filename)
            #chl_filename = strops.ireplace(in_dir, "output", chl_filename)
            chl_filename = filename.replace(unit.upper(), 'CHL')
            chl_filename = chl_filename.replace(in_dir, 'output')
            dir_name = os.path.dirname(chl_filename)
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

            if os.path.exists(chl_filename):
                if not replace:
                    continue
                else:
                    os.remove(chl_filename)
            try:
                chl_with_geo(filename, chl_filename=chl_filename)
                print("Generated chl for %s" % filename)
                time.sleep(1)
            except ValueError:
                print("Error generating chl for %s" % filename)


def chl_with_geo(image_filename, chl_filename=None, mask_val=-1,
                 save_uint=False):
    """
    Takes a registered or registered masked IDS file, reads the data and 
    geo metadata, and writes a new file with chl information and 
    the geo metadata.
    NOTE: it's a simplified version of 
    improc.cameras.tetracam.ndvi_with_geo and cal_ndvi    
    
    Parameters
    ----------
    image_filename: str
        Filename of the IDS image file
    chl_filename: str (opt)
        Output filename for the chl file 
    Returns
    -------
    chl_image: 2darray
        chl image calculated in the program. 
    """


    image = imio.imread(image_filename)
    
    spectral_axis = imio.guess_spectral_axis(image) 
    if spectral_axis == 0:
        image = imio.axshuffle(image)
    
    #NIR is alwasy the 1st band, 550 can be 3rd or 4th
    if image.shape[2] == 4 or image.shape[2] == 5:
        imf_grn = image[:, :, 3].astype('float32')     
        imf_nir = image[:, :, 0].astype('float32')  
    elif image.shape[2] == 3:
        imf_grn = image[:, :, 2].astype('float32')     
        imf_nir = image[:, :, 0].astype('float32')   
    else:
        raise ValueError("Image dimensions do not appear to be correct.")
        
    imf_nir = np.ma.masked_equal(imf_nir, 0)
    chl_image = (imf_nir / imf_grn) - 1.0
    chl_image[chl_image.mask] = mask_val
    chl_image = chl_image.astype('float32')
    
    rastertools.write_geotiff_with_source(image_filename, chl_image,
            chl_filename, nodata=-1, compress=False)

    return chl_image
    

def percent_plot(input_file, bg_value=None, auto_acre=None, auto_acre_new=None,
        l_value=None, soil_value=0.4, l_bound=5, u_bound=90, num_bins=5):
    """        
    creates a histogram like plot that reports acreage of certain NDVI values
    by default it plots between 5% percentile and max NDVI values
    but can be changed by adjust some input parameters
    it mostly like will only work with NDVI derived from the standard procedure 
    background/masked value should be set to NaN or -1
    
    !!! acreage report currently not working with sub-fields !!!
    
    this version takes soil out the does histogram based on statistics. should
    not have problems like l_value smaller than soil value and etc like the 
    "old verison" above
    
    Parameters
    ----------
    input_file: str
        full path of NDVI image
    bg_value: float
        pixel value of background, needs to be set if it's not -1 or NaN
    auto_acre: int
        default is the auto_acre from Fields.csv of database
    l_value: float
        lower end of NDVI value of the plot
    soil_value: float
        NDVI that's smaller than this value is considered soil / non-veg
    l_bound: int
        lower end of NDVI value but expressed as percentile (5 means 5%)
    u_bound: int
        upper end of NDVI value but expressed as percentile (90 means 90%)
    num_bins: int
        number of bars in the chart
    """
    
    img = imio.imread(input_file)
    img[np.isnan(img)] = -1
    if bg_value is not None:
        img[img==bg_value] = -1
    masked = np.ma.masked_less_equal(img, -1)    # masked out background
    values = classify.prep_im(masked)
    
    fl = input_file.lower()
    if ("chl" in fl and soil_value==0.4):       # this is a little problematic
        soil_value = 1.5
        
    # find total acreage
    if auto_acre is None:
        auto_acre = get_acreage_from_filename(input_file)
    # find total acreage minus soil
    if auto_acre_new is None:
        auto_acre_new = auto_acre*(1-stats.percentileofscore(values,soil_value)
                                   /100.0)
    
    masked = np.ma.masked_less(masked, soil_value)  # furhter mask out all non-veg pixels
    values = classify.prep_im(masked)               # veg-only pixels
    
    max_value = np.max(values)
    u_value = stats.scoreatpercentile(values, u_bound)
    if l_value is None:
        l_value = stats.scoreatpercentile(values, l_bound)
        
    if soil_value > l_value:
        sys.exit("ERROR: lower value smaller than soil Value")
    elif l_value > u_value:
        sys.exit("ERROR: upper value smaller than lower value")
        #print("ERROR: upper value smaller than lower Value")
        #return
    
    slices = np.empty(num_bins+2,dtype=float)  
    slices[0] = soil_value
    slices[-1] = max_value        
    slices[1:-1] = np.linspace(l_value, u_value, num_bins)
    heights = np.empty(slices.shape, slices.dtype)
                
    for i in range(len(slices)):
        heights[i] = stats.percentileofscore(values, slices[i]) / 100.0
        
    y = np.empty(heights.shape, heights.dtype)
    #y[0] = heights[0]
    for i in range(len(heights)-1):
        y[i] = heights[i+1]-heights[i]
            
    # find total acreage
    y = y * auto_acre_new

    print os.path.basename(input_file)
    print "!!!auto acreage does NOT work with subfields!!!"
    print auto_acre, ' / ' ,auto_acre_new, ' / ' ,sum(y[1:-1])
    print slices
    print heights
    print y
    
    fig = plt.figure()
    ax = plt.gca()
    rects = ax.bar(slices[1:-1], y[1:-1], 
                   width=(slices[2]-slices[1])*0.9, color='green')
    if num_bins == 5:
        if ("ndvi" in fl):
            rects[0].set_color([0.84,0.97,0.88])
            rects[1].set_color([0.7,0.88,0.67])
            rects[2].set_color([0.45,0.77,0.42])
            rects[3].set_color([0.14,0.55,0.11])           
            ax.set_ylabel('Acres')
            ax.set_xlabel('NDVI')
        elif ("chl" in fl):
            rects[0].set_color([0.84,0.19,0.15])
            rects[1].set_color([0.99,0.68,0.38])
            rects[2].set_color([1,1,0.75])
            rects[3].set_color([0.65,0.85,0.42])           
            ax.set_ylabel('Acres')
            ax.set_xlabel('CHL')
        
    # attach text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height,
                '%.1f'%height, ha='center', va='bottom')

    # set x axis labels
    ax.set_xticks(slices[1:-1]+(slices[2]-slices[1])*0.45)
    x_label = ['' for x in range(num_bins)]
    for i in range(len(slices)-2):
        x_label[i] = '%.2f'%slices[i+1] + '-' + '%.2f'%slices[i+2]
    ax.set_xticklabels(x_label)
    
    #plt.tight_laybout()
    fig.canvas.draw()
    fig.show()         
        

def get_acreage_from_filename(filename):
    """
    simple function that utilizes the improc infrastructure to
    find acreage based on an input filename
    """
    
    fid = finder.get_fid_from_filename(filename)
    
    db = loader.LoadDb()
    fields = db.tables['Fields']
    auto_acre = fields.loc[fid].Auto_acres
    
    return auto_acre
    
    
def get_sub_acreage(filename):
    """
    simple function that prints out subfield names (from mosaic column)
    and acreage of a given shapefile
    """
    
    acres_per_sq_deg = 2471050
    with fiona.open(filename) as polygons:
        areas = [shapely.geometry.shape(p["geometry"]).area for p in polygons]
        sub_ids = [p['properties']['mosaic'] for p in polygons]
    unq_ids = np.unique(sub_ids)
    areas = np.array(areas) * acres_per_sq_deg
    unq_areas = np.empty(len(unq_ids), dtype='float')
    for i in range(len(unq_ids)):
        for j in range(len(sub_ids)):
            if unq_ids[i] == sub_ids[j]:
                unq_areas[i]=unq_areas[i]+float(areas[j])
        print unq_ids[i], unq_areas[i]         


def extract_points(indx_files, shp_file, csv_file, use_local=False,
                         find_min=False, mask_val=0):
    """        
    extract index values at sample locations from one or more veg index images 
    sample locations defined by a shapefile (points)
    it's kind of usefule but also limited
    currently only works for single band index image
    not really recommned using local max/min function
    and be aware of needing to find loc max for GCI but loc min for TCARI
    
    Parameters
    ----------
    indx_files: str
        full path of vegetation index image
    shp_file: str
        full path of a shapfile of areas of interests (points)
    csv_file: str
        output results to .csv file
    use_local: bool (opt)
        option to find local max/min, default to FALSE
    find_min: bool (opt)
        parameter needed in classify.local_extrema function 
    mask_val: int (opt)
        mask value used when finding local max/min, default to 0
    """
    
    output = shapetools.dataframe_from_shapefile(shp_file)
    
    for f in indx_files:
        #if use_local is False:
        if  not use_local:  # no searching local max/min
            df = extract.match_points(f, shp_file)
        else:
            local_ext = functools.partial(
                classify.local_extrema, find_min=find_min, mask_val=mask_val)
            df = extract.match_points(f, shp_file, mask_func=local_ext)
        output[os.path.basename(f)] = df['dn']

    output.to_csv(csv_file, index=False, sep=',')
    
    return output
        

#==============================================================================
# this section is related to classification on chl / NDVI image
# including different ways (loc max / mean) and counting class for reporting

def count_class(image_filename, plots=True,
                colors=[[255,0,0],[255,255,0],[0,135,14],[0,0,255],[0,0,0]]):
    """
    count the pixels of each of the class. report counts, percentage and 
    a summary bar chart
    
    Parameters
    ----------
    image_filename: str
        full path of the classified image file
    plots: bool
        show a bar chart or not
    colors: numpy array
        rgb code of the classes followed by black [0,0,0] 
        examples of color code:
        red: [255,0,0]
        yellow: [255,255,0]
        green: [0,135,14]
        blue: [0,0,255]
        black: [0, 0, 0]
        
        red: [253,141,60]
        yellow: [255,255,0]
        green: [116,196,118]
        blue: [35,67,132]
        black: [0, 0, 0]
        
        red: [255, 0, 0]
        orange: [255, 146, 0]
        yellow: [255, 255, 0]
        green: [0, 135, 14]
        blue: [0, 0, 255]
        black: [0, 0, 0]
    """
    
    img = imio.imread(image_filename)
    spectral_axis = imio.guess_spectral_axis(img) 
    if spectral_axis == 0:
        img = imio.axshuffle(img)
    
    colors = np.asarray(colors)
    counts = np.zeros(colors.shape[0], dtype='uint')
    heights = np.zeros(counts.shape[0]-1, dtype='float32')  # no plotting black color

    for i in range(len(counts)):
        tmp = ((img[:,:,0]==colors[i,0])*(img[:,:,1]==colors[i,1])*
               (img[:,:,2]==colors[i,2]))
        counts[i] = len(tmp[tmp])
        
    for i in range(len(counts)-1):
        heights[i] = float(counts[i])/float(sum(counts[:-1]))
        print "color%s: %s, %.4f" %(i, counts[i], heights[i])
           
    if img.shape[0]*img.shape[1] != sum(counts):
        print "sum don't match"
        
    if plots:
        fig = plt.figure()
        ax = plt.gca()
        rects = ax.bar(np.arange(len(heights)), heights, width=0.9)
        for i in range(len(heights)):
            rects[i].set_color(colors[i,:].astype('float32')/255.)               
        ax.set_ylabel('percent')
        ax.set_xlabel('class')
        ax.set_xticks(np.arange(len(heights))+0.45)
        ax.set_xticklabels(np.arange(len(heights)))        
        

def chl_classi_loc_mean(chl_file, bkg_thres=0.75, ndvi=False,
        loc_mean_file=None, uniform_file=None, max_file=None, adapt_thres=11,
        selem_size=4, min_distance=3, radius=7, labels=False, small_thres=3,
        footprint=True, manual_find=False, use_uniform=True, quicklook=False):
    """
    sandbox area for chl image classification
    this version has skimage local mean feature in it
    options to use watershed segmentation (label) or not
    options to manually find local max with label
    options to use uniform_tree to fill segment or not
    
    Parameters
    ----------
    chl_file: str
        full path of input chl file
    bkg_thres: float
        threshold for masking out soil background and other stuff
    ndvi: bool
        False if used on chl files. It's used to set up output filenames
        the algorithm is the same
    loc_mean_file: str
        full path of output local mean file. Pass desired filename if you don't
        want to use default filename
    uniform_file: str
        full path of output uniform file
    max_file: str
        full path of output max file
    adapt_thres: int
        block size for adaptive threshold function
    selem_size: int
        define the size of the neighborhood for calculating local mean
    min_distance: int
        minimum distance between two local peak values
    radius: int
        size to expand local max value
    labels: bool
        optional segmentation file used when finding local max
    small_thres: in
        threshold used to get rid of small objects
    footprint: bool
        if True, then footprint=selem will be used for searching local max
    manual_find: bool
        if True, after watershed segmentation, local max will be manually found
        instead of using skimage.feature.peak_local_max
    use_uniform: bool
        if True, after manually finds local max, improc.classify.uniform_tree 
        will be used to fill the blocks. if False, segment will be filled manually
    quicklooks:
        if True, histogram of uniform file will be displayed, after entering 
        lower and upper limit, a quicklook of colored uniform file will be displayed
    """
        
    start_time = time.time()
    
    # preparation of output filenames    
    if loc_mean_file is None:
        loc_mean_file = chl_file.replace('output', 'color')
        if ndvi:
            loc_mean_file = loc_mean_file.replace('NDVI', 'NDVI_loc_mean')
        else:
            loc_mean_file = loc_mean_file.replace('chl', 'chl_loc_mean')
    else:
        loc_mean_file = loc_mean_file
        
    if uniform_file is None:
        uniform_file = loc_mean_file.replace('loc_mean', 'uniform')
    else:
        uniform_file = uniform_file
        
    if max_file is None:
        max_file = uniform_file.replace('uniform', 'max')
    else:
        max_file = max_file
        
    log_file = max_file.replace('max', 'log')
    log_file = log_file.replace('tif', 'txt')   # issue for non-tiff images
    
    dir_name = os.path.dirname(uniform_file)    # make "color" folder if it does not exist
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    
    # processing for image enhancement
    chl_img = imio.imread(chl_file)
    chl_img = np.nan_to_num(chl_img)    
    chl_img = np.ma.masked_less_equal(chl_img, bkg_thres)
    chl_img[chl_img.mask] = 0.0
    chl_img = np.uint16(chl_img*1000)
    
    selem = skimage.morphology.disk(selem_size)
    loc_mean = skimage.filters.rank.mean(chl_img, selem=selem,
                                         mask=~chl_img.mask)
    loc_mean = np.float32(loc_mean)/1000.0
    #loc_mean[chl_img.mask] = 0.0
    rastertools.write_geotiff_with_source(chl_file, loc_mean, loc_mean_file,
                                          nodata=0, compress=False)  
    #perc_mean = skimage.filters.rank.mean_percentile(chl_img, selem=selem,p0=.1, p1=.9)
    #perc_mean = np.float32(perc_mean)/1000.0
    #rastertools.write_geotiff_with_source(chl_file, perc_mean, perc_mean_file, nodata=0, compress=False)
    
    if footprint:   # if use footprint, it'll replace min_distance in peak_local_max
        footprint = selem
        footprintYN = True
    else:
        footprint = None
        footprintYN = False
   
    bw = skimage.filters.threshold_adaptive(loc_mean, adapt_thres)

    # use watershed segmentation (label) or not
    if labels == False:
        labelsYN = False
        small_thres = False
        chl_img_max=feature.peak_local_max(loc_mean, min_distance=min_distance,                                                 
                                           exclude_border=True, indices=False,
                                           footprint=footprint, labels=bw)
        uniform = classify.uniform_trees(loc_mean, chl_img_max, radius=radius)
        use_uniform = True
    else:
        labelsYN = True        
        distance = scipy.ndimage.distance_transform_edt(bw)
        loc_max = feature.peak_local_max(distance, indices=False,
                                         min_distance=min_distance, labels=bw)
        markers = scipy.ndimage.label(loc_max)[0]
        labels = skimage.morphology.watershed(-distance, markers, mask=bw)
        label_file = loc_mean_file.replace('loc_mean', 'label')
        
        if small_thres is None:
            labels = labels
        else:
            small_selem = np.ones((small_thres, small_thres))
            small_obj = morphology.binary_opening(labels, selem=small_selem)
            labels[~small_obj]=0  

        rastertools.write_geotiff_with_source(chl_file, labels, label_file,
                                              nodata=0, compress=False)    
    
        if manual_find == False:    # this usually runs into problems with lots of segment
            chl_img_max = feature.peak_local_max(
                loc_mean, min_distance=1, exclude_border=False, indices=False,
                footprint=footprint, labels=labels)
            uniform = classify.uniform_trees(loc_mean, chl_img_max,
                                             radius=radius)
        else:   # this sectoin manaully finds local max
            statCC = skimage.measure.regionprops(labels)
            uniform = np.zeros(chl_img.shape, dtype=loc_mean.dtype)
            chl_img_max = np.zeros(chl_img.shape, dtype=bool)
            #data = np.empty(len(statCC), dtype=loc_mean.dtype)
    
            for i in range(len(statCC)):    # there may be a better way to do this part
                bb = statCC[i].bbox
                x1 = bb[0]; y1 = bb[1]; x2 = bb[2]; y2 = bb[3]
                if statCC[i].area > small_thres**2:
                    uniform[x1:x2, y1:y2] = np.max(loc_mean[x1:x2, y1:y2])
                    tmp = np.ma.masked_equal(loc_mean[x1:x2, y1:y2],
                                             np.max(loc_mean[x1:x2, y1:y2]))
                    chl_img_max[x1:x2, y1:y2][tmp.mask] = 'True'                         
                    #for j in range(x1, x2):
                    #    for k in range(y1, y2):
                    #        if loc_mean[j,k] == np.max(loc_mean[x1:x2, y1:y2]):
                    #            chl_img_max[j,k] = 'True'
                else:
                    uniform[x1:x2, y1:y2] = 0
                
            if use_uniform == False:
                uniform = uniform
            else:
                uniform = classify.uniform_trees(loc_mean, chl_img_max,
                                                 radius=radius)                                             
    
    uniform[chl_img.mask] = 0.0
    rastertools.write_geotiff_with_source(chl_file, uniform, uniform_file,
                                          nodata=0, compress=False)
    loc_mean[~chl_img_max] = float('NaN')
    rastertools.write_geotiff_with_source(chl_file, loc_mean, max_file,
                                          nodata=0, compress=False)

    max_of_max = np.max(loc_mean[chl_img_max])
    mean_of_max = np.mean(loc_mean[chl_img_max])
    std_of_max = np.std(loc_mean[chl_img_max])
    print     
    print "max: %.4f" %(max_of_max)
    print "mean: %.4f" %(mean_of_max)
    print "std dev: %.4f" %(std_of_max)
    print "5 bins:"
    print "%.4f, %.4f" %(mean_of_max*0.5, mean_of_max*0.8)
    print "%.4f, %.4f" %(mean_of_max*0.8, mean_of_max*0.9)
    print "%.4f, %.4f" %(mean_of_max*0.9, mean_of_max*1.1)
    print "%.4f, %.4f" %(mean_of_max*1.1, mean_of_max*1.2)
    print "%.4f, %.4f" %(mean_of_max*1.2, mean_of_max*1.5)      
    
    log = open(log_file, 'w')
    log.write(" background_threshold=%s\n" %(bkg_thres))
    log.write(" adaptive_threshold=%s\n" %(adapt_thres))    
    log.write(" selem_size=%s\n min_distance=%s\n" %(selem_size,min_distance)) 
    log.write(" footprint=%s\n use_uniform=%s\n" %(footprintYN, use_uniform))
    log.write(" radius=%s\n" %(radius))
    log.write(" labels=%s\n small_thres=%s\n" %(labelsYN, small_thres))
    log.write(" manually_find_max=%s\n" %(manual_find))
    log.write(" max_value=%.4f\n" %(max_of_max))
    log.write(" mean_value=%.4f\n" %(mean_of_max))
    log.write(" std_dev=%.4f\n" %(std_of_max))
    log.write(" %s seconds" %(time.time() - start_time))    
    log.close()    
    
    #show quicklooks or not    
    if quicklook == True:
        uniform = np.nan_to_num(uniform)
    
        fig = plt.figure()
        ax = plt.gca()
        ax.hist(uniform.flatten(), bins=100)
        fig.canvas.draw()
        fig.show()
    
        l_lim = raw_input('enter lower limit:')
        u_lim = raw_input('enter upper limit:')
        fig = plt.figure()
        imgplot = plt.imshow(uniform, cmap='spectral_r')
        imgplot.set_clim(l_lim, u_lim)
        #plt.colorbar()    

    print "--- %.2f seconds ---" %(time.time() - start_time)
    print


def chl_classi_loc_max(chl_file, ndvi=True, uniform_file=None, max_file=None,
                       bkg_thres=0.4, adapt_thres=11, footprint=None, radius=7,
                       min_distance=3, labels=False, small_thres=None):
    """
    sandbox area for chl image classification
    this version uses skimage local max feature
    
    Parameters
    ----------
    chl_file: str
        fulll path of input chl file
    uniform_file: str
        full path of output uniform file
    max_file: str
        fulll path of output local maximum file
    radius: int
        size to expand local maximum
    min_distance: int
        minimum distance between two local maximum
    labels: bool
        optional segmentation file used when finding local maximum
    small_thres: int
        threshold for getting rid of small objects
    """
    
    # preparation of output filenames
    if uniform_file is None:
        uniform_file = chl_file.replace('output', 'color/loc_max')
        if ndvi:
            uniform_file = uniform_file.replace('NDVI', 'NDVI_uniform')
        else:
            uniform_file = uniform_file.replace('chl', 'chl_uniform')
    else:
        uniform_file = uniform_file
    
    if max_file is None:
        max_file = uniform_file.replace('uniform', 'max')
    else:
        max_file = max_file
    
    log_file = uniform_file.replace('uniform', 'log')
    log_file = log_file.replace('tif', 'txt')
    
    dir_name = os.path.dirname(uniform_file)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    
    # processing for image enhancement
    #ndvi_img = improc.imops.imio.imread(ndvi_files[1])
    chl_img = imio.imread(chl_file)
    #ndvi_img = np.nan_to_num(ndvi_img)
    chl_img = np.nan_to_num(chl_img)
    chl_img = np.ma.masked_less_equal(chl_img, bkg_thres)
    chl_img[chl_img.mask] = 0
    #ndvi_img = improc.tests.anisodiff2y3d.anisodiff(ndvi_img, niter=3, kappa=80, gamma=0.2)
    tmp_img = anisodiff2y3d.anisodiff(chl_img, niter=3, kappa=80, gamma=0.2)
    
    if footprint is None:   # if use footprint, it'll replace min_distance in peak_local_max
        footprintYN = False
    else:
        footprintYN = True
        footprint = skimage.morphology.disk(footprint)        
        
    bw = skimage.filters.threshold_adaptive(tmp_img, adapt_thres)
    
    if labels == False:
        labelsYN = False
        chl_img_max = feature.peak_local_max(
            tmp_img, min_distance=min_distance, exclude_border=True,
            indices=False, footprint=footprint, labels=bw)
    else:   # usually runs into problems with large numbers of segment
        labelsYN = True
        distance = scipy.ndimage.distance_transform_edt(bw)
        loc_max = feature.peak_local_max(distance, indices=False,
                                         min_distance=min_distance, labels=bw)
        markers = scipy.ndimage.label(loc_max)[0]
        labels = skimage.morphology.watershed(-distance, markers, mask=bw)
                
        if small_thres is None:
            labels = labels
        else:
            small_selem = np.ones((small_thres, small_thres))
            small_obj = skimage.morphology.binary_opening(labels,
                                                          selem=small_selem)
            labels[~small_obj] = 0
        
        label_file = uniform_file.replace('uniform', 'label')
        rastertools.write_geotiff_with_source(chl_file, labels, label_file,
                                              nodata=0, compress=False)    
        chl_img_max = feature.peak_local_max(
            tmp_img, min_distance=1, exclude_border=False, indices=False,
            footprint=footprint, labels=labels)
                                                         
    uniform = classify.uniform_trees(chl_img, chl_img_max, radius=radius)
    uniform[chl_img.mask] = 0
    rastertools.write_geotiff_with_source(chl_file, uniform, uniform_file,
                                          nodata=-1, compress=False)
    chl_img[~chl_img_max] = float('NaN')
    rastertools.write_geotiff_with_source(chl_file, chl_img, max_file,
                                          nodata=-1, compress=False)
                                          
    max_of_max = np.max(chl_img[chl_img_max])
    mean_of_max = np.mean(chl_img[chl_img_max])
    std_of_max = np.std(chl_img[chl_img_max])
    print "max: ", max_of_max
    print "mean: ", mean_of_max
    print "std dev: ", std_of_max
    print "5 bins: "
    print mean_of_max*0.5, mean_of_max*0.8
    print mean_of_max*0.8, mean_of_max*0.9
    print mean_of_max*0.9, mean_of_max*1.1
    print mean_of_max*1.1, mean_of_max*1.2
    print mean_of_max*1.2, mean_of_max*1.5
    print
    
    log = open(log_file, 'w')
    log.write(" background_threshold=%s\n" %(bkg_thres))
    log.write(" adaptive_threshold=%s\n" %(adapt_thres))
    log.write(" min_distance=%s\n radius=%s\n" %(min_distance, radius))
    log.write(" labels=%s\n small_thres=%s\n" %(labelsYN, small_thres))
    log.write(" footprint=%s\n" %(footprintYN))
    log.write(" max_value=%s\n" %(max_of_max))
    log.write(" mean_value=%s\n" %(mean_of_max))
    log.write(" std_dev=%s\n" %(std_of_max))
    log.close() 
                                         

def colorize_chl_classi(uniform_file, max_file, num_classes, slices_ext=None):
    """
    Given an chl classification image, makes a colored version
    this function was a modification from improc.postprocess.colorize_visnir
    Parameters
    ----------
    uniform_filename: str
        Full path to file to recolor
    num_classes: int
        How many classes in the image
    slices_ext: list
        Pass in your own list of boundaries between slice colors.
    """
    
    if num_classes == 4:
        colors = [[255, 0, 0], [255, 255, 0], [0, 135, 14],
                  [0, 0, 255]]
    elif num_classes == 5:
        colors = [[255, 0, 0], [255, 146, 0], [255, 255, 0],
                  [0, 135, 14], [0, 0, 255]]
    
    im = imio.imread(uniform_file)
    out_im = np.zeros(im.shape + (3,), dtype='uint8')
    #im[np.isnan(im)] = -1
    
    mx = imio.imread(max_file)
    min_mx = np.min(mx[~np.isnan(mx)]) - 0.001
    max_mx = np.max(mx[~np.isnan(mx)]) + 0.001
    mean_mx = np.mean(mx[~np.isnan(mx)])
    std_mx = np.std(mx[~np.isnan(mx)])
    
    if slices_ext is None:
        if num_classes == 4:
            slices_ext = np.array([min_mx, mean_mx-1.5*std_mx,
                                   mean_mx-0.5*std_mx, mean_mx+std_mx, max_mx])
        if num_classes == 5:          
            slices_ext = np.array([min_mx, mean_mx*0.8, mean_mx*0.9,
                                   mean_mx*1.1, mean_mx*1.2, max_mx])
    else:
        slices_ext = slices_ext
        
    print slices_ext    
    
    lowers = slices_ext[:-1]
    uppers = slices_ext[1:]
    rgb_channels = np.zeros(lowers.shape + (3,), dtype='uint8')

    for i in range(len(colors)):
        rgb_channels[i, :] = colors[i]

    for i, (l, u) in enumerate(zip(lowers, uppers)):
        in_range = np.logical_and(im <= u, im > l)
        out_im[:, :, 0][in_range] = rgb_channels[i, 0]
        out_im[:, :, 1][in_range] = rgb_channels[i, 1]
        out_im[:, :, 2][in_range] = rgb_channels[i, 2]

    return out_im


def geo_colorize_chl_classi(num_classes, chl_file=None, ndvi=False,
                            loc_mean=True, uniform_file=None, max_file=None,
                            out_filename=None, slices_ext=None):
    """
    Wrapper for colorize_chl_classi
    """
    
    if chl_file is None:
        uniform_file = uniform_file
        max_file = max_file
    elif loc_mean:
        uniform_file = chl_file.replace('output', 'color')
        if ndvi:
            uniform_file = uniform_file.replace('NDVI', 'NDVI_uniform')
            max_file = uniform_file.replace('uniform', 'max')
        else:
            uniform_file = uniform_file.replace('chl', 'chl_uniform')
            max_file = uniform_file.replace('uniform', 'max')
    else:
        uniform_file = chl_file.replace('output', 'color/loc_max')
        if ndvi:
            uniform_file = uniform_file.replace('NDVI', 'NDVI_uniform')
            max_file = uniform_file.replace('uniform', 'max')
        else:
            uniform_file = uniform_file.replace('chl', 'chl_uniform')
            max_file = uniform_file.replace('uniform', 'max')
    
    if os.path.exists(uniform_file) and os.path.exists(max_file):
        out_im = colorize_chl_classi(uniform_file, max_file, num_classes,
                                 slices_ext=slices_ext)
    else:
        sys.exit("file(s) not exist")        
    
    if out_filename is None:
        out_filename = uniform_file.replace('uniform', 'class')
    else:
        out_filename = out_filename
    rastertools.write_geotiff_with_source(uniform_file, out_im, out_filename)
    
    return out_im
    

#==============================================================================
# this section is for old stuff and sandbox

def percent_plot_old(input_file, bg_value = None, auto_acre=None, l_value=None,
                     soil_value=0.4, l_bound=5, u_bound=90, num_bins=5):
    """        
    creates a histogram like plot that reports acreage of certain NDVI values
    by default it plots between 5% percentile and max NDVI values
    but can be changed by adjust some input parameters
    it mostly like will only work with NDVI derived from the standard procedure 
    background/masked value should be set to NaN or -1
    
    !!! acreage report currently not working with sub-fields !!!
    
    Parameters
    ----------
    input_file: str
        full path of NDVI image
    bg_value: float
        pixel value of background, needs to be set if it's not -1 or NaN
    auto_acre: int
        default is the auto_acre from Fields.csv of database
    l_value: float
        lower end of NDVI value of the plot
    soil_value: float
        NDVI that's smaller than this value is considered soil / non-veg
    l_bound: int
        lower end of NDVI value but expressed as percentile (5 means 5%)
    u_bound: int
        upper end of NDVI value but expressed as percentile (90 means 90%)
    num_bins: int
        number of bars in the chart
    """
    
    img = imio.imread(input_file)
    img[np.isnan(img)] = -1
    if bg_value is not None:
        img[img==bg_value] = -1
    masked = np.ma.masked_less_equal(img, -1)
    values = classify.prep_im(masked)
    
    max_value = np.max(values)
    #bins, percentiles = slicing.get_percentiles(values)
    u_value = stats.scoreatpercentile(values, u_bound)
    if l_value is None:
        l_value = stats.scoreatpercentile(values, l_bound)
        
    if soil_value>l_value:
        print ("lower boundary is smaller than soil value, check the image")        
    elif l_value>u_value:
        print ("lower boundary is larger than upper boundary, check the image")
    else:
        slices = np.empty(num_bins+2,dtype=float)
        slices[0] = soil_value
        slices[-1] = max_value        
        slices[1:-1] = np.linspace(l_value, u_value, num_bins)
        heights = np.empty(slices.shape, slices.dtype)
                
        for i in range(len(slices)):
            #diff = abs(bins-slices[i])
            heights[i] = stats.percentileofscore(values, slices[i]) / 100.0
        
        y = np.empty(heights.shape, heights.dtype)
        #y[0] = heights[0]
        for i in range(len(heights)-1):
            y[i] = heights[i+1]-heights[i]
            
        # find total acreage
        if auto_acre is None:
            auto_acre = get_acreage_from_filename(input_file)
        y = y * auto_acre

        print auto_acre, ' / ' ,sum(y[1:-1])
        print slices
        print heights
        print y
        
        fig = plt.figure()
        ax = plt.gca()
        rects = ax.bar(slices[1:-1], y[1:-1], 
                       width=(slices[2]-slices[1])*0.9, color='green')
        ax.set_ylabel('Acres')
        ax.set_xlabel('NDVI')
        
        # attach text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height,
                    '%.1f'%height, ha='center', va='bottom')

        # set x axis labels
        ax.set_xticks(slices[1:-1]+(slices[2]-slices[1])*0.45)
        x_label = ['' for x in range(num_bins)]
        for i in range(len(slices)-2):
            x_label[i] = '%.2f'%slices[i+1] + '-' + '%.2f'%slices[i+2]
        ax.set_xticklabels(x_label)
        
        #plt.tight_laybout()
        fig.canvas.draw()
        fig.show()         
        

    #some previously tried ways to count
    #not really useful, just for record keeping    
    #bool_blue = (img[:,:,0]==0) * (img[:,:,1]==0) * (img[:,:,2]==255)
    #blue = len(bool_blue[bool_blue])
    
    #bool_green = (img[:,:,0]==0) * (img[:,:,1]==135) * (img[:,:,2]==14)
    #green = len(bool_green[bool_green])
    
    #bool_yellow = (img[:,:,0]==255) * (img[:,:,1]==255) * (img[:,:,2]==0)
    #yellow = len(bool_yellow[bool_yellow])
    
    #bool_red = (img[:,:,0]==255) * (img[:,:,1]==0) * (img[:,:,2]==0)
    #red = len(bool_red[bool_red])
    
    #bool_black = (img[:,:,0]==0) * (img[:,:,1]==0) * (img[:,:,2]==0)
    #black = len(bool_black[bool_black])    
    
    #blue = len(img[img[:,:,2]==255])
    #green = len(img[img[:,:,1]==135])
    #yellow = len(img[img[:,:,1]==255])
    #red = len(img[img[:,:,0]==255]) - yellow
    #black = len(img[img[:,:,0]==0]) - blue - green
    
    #blue=0
    #green=0
    #yellow=0
    #red=0
    #black=0   
    #for i in range(img.shape[0]):
    #    for j in range(img.shape[1]):
    #        if img[i,j,0] == 0 and img[i,j,1] == 0 and img[i,j,2] == 255:
    #            blue = blue + 1
    #        if img[i,j,0] == 0 and img[i,j,1] == 135 and img[i,j,2] == 14:
    #            green = green + 1
    #        if img[i,j,0] == 255 and img[i,j,1] == 255 and img[i,j,2] == 0:
    #            yellow = yellow + 1
    #        if img[i,j,0] == 255 and img[i,j,1] == 0 and img[i,j,2] == 0:
    #            red = red + 1
    #        if img[i,j,0] == 0 and img[i,j,1] == 0 and img[i,j,2] == 0:
    #            black = black + 1
                
    
    #blue: (35,67,132)
    #green: (116,196,118)
    #yellow: (255,255,0)
    #red: (253,141,60)
            
    #blue = len(img[img[:,:,0]==35])
    #green = len(img[img[:,:,0]==116])
    #yellow = len(img[img[:,:,0]==255])
    #red = len(img[img[:,:,0]==253])
    #black = len(img[img[:,:,0]==0])
    
    #fig = plt.figure()
    #ax = plt.gca()
    #rects = ax.bar([1,2,3,4],
    #               [float(red)/float(blue+green+yellow+red),
    #                float(yellow)/float(blue+green+yellow+red),
    #                float(green)/float(blue+green+yellow+red),
    #                float(blue)/float(blue+green+yellow+red)], width=0.9)
    #rects[0].set_color([0.99,0.55,0.24])
    #rects[1].set_color([1,1,0.2])
    #rects[2].set_color([0.45,0.77,0.46])
    #rects[3].set_color([0.14,0.26,0.52])           
    #ax.set_ylabel('percent')
    #ax.set_xlabel('class')    
    
 
def chl_classi_matlab(chl_file, bkg_thres=0.4, meanYN=True, ndvi=False,
                      uniform_file=None, max_file=None, radius=7,
                      min_distance=3, labels=False, small_thres=3):
    """
    sandbox area for chl image classification
    this version is to mimic Yibin's matlab program
    
    Parameters
    ----------
    chl_file: str
        fulll path of input chl file
    bkg_thres: float
        soil or background value to be masked out
    meanYN: bool
        if True, then calculate local mean; if False, then calculate local max
    uniform_file: str
        full path of output uniform file
    max_file: str
        fulll path of output local maximum file
    radius: int
        size to expand local maximum
    min_distance: int
        minimum distance between two local maximum
    labels: bool
        optional segmentation file used when finding local maximum
    small_thres: int
        threshold for getting rid of small objects
    """

    chl_img = imio.imread(chl_file)
    chl_img = np.nan_to_num(chl_img)    
    chl_img = np.ma.masked_less_equal(chl_img, bkg_thres)
    chl_img[chl_img.mask] = 0
        
    # perparation of output filenames
    if meanYN:
        label_file = chl_file.replace('output', 'color')
    else:
        label_file = chl_file.replace('output', 'color/loc_max')
        
    if ndvi:
        label_file = label_file.replace('NDVI', 'NDVI_label')
    else:
        label_file = label_file.replace('chl', 'chl_label')
        
    if uniform_file is None:
        uniform_file = label_file.replace('label', 'unifrom')
    else:
        uniform_file = uniform_file
    
    if max_file is None:
        max_file = label_file.replace('label', 'max')
    else:
        max_file = max_file    
    
    # image transformation and segmentation
    tmp_img = anisodiff2y3d.anisodiff(chl_img, niter=3, kappa=80, gamma=0.2)
    tmp_img[chl_img.mask]=0
    bw = skimage.filters.threshold_adaptive(tmp_img, 11)
    distance = scipy.ndimage.distance_transform_edt(bw)
    loc_max = skimage.feature.peak_local_max(distance, indices=False,
                                                 min_distance=min_distance,
                                                 labels=bw)
    markers = scipy.ndimage.label(loc_max)[0]
    labels = skimage.morphology.watershed(-distance, markers, mask=bw)
    rastertools.write_geotiff_with_source(chl_file, labels, label_file,
                                          nodata=0, compress=False)
    chl_img1 = copy.deepcopy(chl_img)
    chl_img1[~loc_max] = float('NaN')
    rastertools.write_geotiff_with_source(chl_file, chl_img1, max_file,
                                          nodata=0, compress=False)
                                          
    # calculating local mean/max 
    small_selem = np.ones((small_thres, small_thres))
    small_obj = skimage.morphology.binary_opening(labels, selem=small_selem)
    labels[~small_obj] = 0
        
    statCC = skimage.measure.regionprops(labels)
    uniform = np.zeros(chl_img.shape, dtype=chl_img.dtype)
    data = np.empty(len(statCC), dtype=chl_img.dtype)
    
    for i in range(len(statCC)):
        bb = statCC[i].bbox
        if meanYN == True:   # local mean
            uniform[bb[0]:bb[2], bb[1]:bb[3]] = np.mean(chl_img[bb[0]:bb[2], bb[1]:bb[3]])
        else:               # local max
            uniform[bb[0]:bb[2], bb[1]:bb[3]] = np.max(chl_img[bb[0]:bb[2], bb[1]:bb[3]])
            
    rastertools.write_geotiff_with_source(chl_file, uniform, uniform_file,
                                          nodata=0, compress=False)
        
    
    # connected component analysis
    
    #numCC = len(statCC)
    #vecArea = np.zeros((numCC,1))
    #vecWdth = np.zeros((numCC, 1))
    #vecHght = np.zeros((numCC, 1))
    #vecCntr = np.zeros((numCC, 1))
    #vecAsp = np.zeros((numCC, 1))
    #vecExt = np.zeros((numCC, 1))
    #vecBB = np.zeros((numCC, 4)) 

    #for i in range(numCC):
    #    vecBB[i, 4] = statCC[i].bbox
    #    vecWdth[i] = vecBB[2] - vecBB[0] + 1
    #    vecHght[i] = vecBB[3] - vecBB[1] + 1
    #    vecArea[i] = statCC[i].area
    #    vecCntr[i] = statCC[i].centroid
    #    vecAsp[i] = vecWdth[i] / vecHght[i]
    #    vecExt[i] = statCC[i].extent
    
