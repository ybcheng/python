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
import pandas as pd
import numpy as np
import time
import copy
from scipy import stats

from ..imops import imio
from ..gis import rastertools
from ..gen import strops


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
                                              
                                              
def gen_chl_files(filenames, in_dir, dummy=None, replace=True):
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
    """
    
    
    for filename in filenames:
        if (filename.endswith(".tif") and ('ids' in filename.lower())):
            # generate a good output filename
            #chl_filename = strops.ireplace("IDS", "chl", filename)
            #chl_filename = strops.ireplace(in_dir, "output", chl_filename)
            chl_filename = filename.replace('IDS', 'chl')
            chl_filename = chl_filename.replace(in_dir, 'otuput')
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
        
