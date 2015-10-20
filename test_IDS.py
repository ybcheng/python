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
import matplotlib.pyplot as plt
from scipy import stats

from ..imops import imio
from ..gis import rastertools
from ..gen import strops
from ..cv import classify, slicing
from ..dbops import finder, loader
from . import anisodiff2y3d


#==============================================================================
# this section is mostly utilities for preprocessing and etc
# various image manipulation functions and empirical line and IARR

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
        
    rastertools.write_geotiff_with_source(filename, iarr,
            iarr_filename, nodata=-1, compress=False)

    return iarr
    

#==============================================================================
# this section is about chl index calculation and a little post processing
# making histogram like bar chart for reporting

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
        

def percent_plot(input_file, bg_value=None, auto_acre=None, auto_acre_new=None,
                 l_value=None, soil_value=0.4, l_bound=5, u_bound=90, 
                 num_bins=5):
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


#==============================================================================
# this section is related to classification on chl / NDVI image
# including different ways (loc max / mean) and counting class for reporting

def count_class(image_filename):
    """
    count the pixels of each of the class:
    blue: (0,0,255)
    green: (0,155,0)
    yellow: (255,255,0)
    red: (255,0,0)
    
    Parameters
    ----------
    image_filename: str
        full path of the classified image file
    """
    
    blue=0
    green=0
    yellow=0
    red=0
    black=0
    
    img = imio.imread(image_filename)
    
    spectral_axis = imio.guess_spectral_axis(img) 
    if spectral_axis == 0:
        img = imio.axshuffle(img)
        
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i,j,0] == 0 and img[i,j,1] == 0 and img[i,j,2] == 255:
                blue = blue + 1
            if img[i,j,0] == 0 and img[i,j,1] == 155 and img[i,j,2] == 0:
                green = green + 1
            if img[i,j,0] == 255 and img[i,j,1] == 255 and img[i,j,2] == 0:
                yellow = yellow + 1
            if img[i,j,0] == 255 and img[i,j,1] == 0 and img[i,j,2] == 0:
                red = red + 1
            if img[i,j,0] == 0 and img[i,j,1] == 0 and img[i,j,2] == 0:
                black = black + 1
                
    print blue, float(blue)/float(blue+green+yellow+red)
    print green, float(green)/float(blue+green+yellow+red)
    print yellow, float(yellow)/float(blue+green+yellow+red)
    print red, float(red)/float(blue+green+yellow+red)
    
    if img.shape[0]*img.shape[1] != blue+green+yellow+red+black:
        print("sum don't match")
        
        
def count_class_new(image_filename):
    """
    same as above but for the color scheme of new classification images
    count the pixels of each of the class:
    blue: (35,67,132)
    green: (116,196,118)
    yellow: (255,255,0)
    red: (253,141,60)
    
    Parameters
    ----------
    image_filename: str
        full path of the classified image file
    """
    
    blue=0
    green=0
    yellow=0
    red=0
    black=0
    
    img = imio.imread(image_filename)
    
    spectral_axis = imio.guess_spectral_axis(img) 
    if spectral_axis == 0:
        img = imio.axshuffle(img)
        
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i,j,0] == 35 and img[i,j,1] == 67 and img[i,j,2] == 132:
                blue = blue + 1
            if img[i,j,0] == 116 and img[i,j,1] == 196 and img[i,j,2] == 118:
                green = green + 1
            if img[i,j,0] == 255 and img[i,j,1] == 255 and img[i,j,2] == 50:
                yellow = yellow + 1
            if img[i,j,0] == 253 and img[i,j,1] == 141 and img[i,j,2] == 60:
                red = red + 1
            if img[i,j,0] == 0 and img[i,j,1] == 0 and img[i,j,2] == 0:
                black = black + 1
                
    print image_filename
    print "red:", red, float(red)/float(blue+green+yellow+red)
    print "yellow:", yellow, float(yellow)/float(blue+green+yellow+red)
    print "green:", green, float(green)/float(blue+green+yellow+red)
    print "blue:", blue, float(blue)/float(blue+green+yellow+red)
    
    if img.shape[0]*img.shape[1] != blue+green+yellow+red+black:
        print("sum don't match")
    
    fig = plt.figure()
    ax = plt.gca()
    rects = ax.bar([1,2,3,4],
                   [float(red)/float(blue+green+yellow+red),
                    float(yellow)/float(blue+green+yellow+red),
                    float(green)/float(blue+green+yellow+red),
                    float(blue)/float(blue+green+yellow+red)], width=0.9)
    rects[0].set_color([0.99,0.55,0.24])
    rects[1].set_color([1,1,0.2])
    rects[2].set_color([0.45,0.77,0.46])
    rects[3].set_color([0.14,0.26,0.52])           
    ax.set_ylabel('percent')
    ax.set_xlabel('class')    
    

def chl_classi_loc_mean(chl_file, bkg_thres=0.4, ndvi=False,
                        loc_mean_file=None, uniform_file=None,
                        max_file=None, selem_size=4, min_distance=3, radius=7,
                        labels=False, small_thres=None, quicklook=False):
    """
    sandbox area for chl image classification
    this version has skimage local mean feature in it
    
    Parameters
    ----------
    chl_file: str
        full path of input chl file
    bkg_thres: float
        threshold for masking out soil background and other stuff
    loc_mean_file: str
        full path of output local mean file. Pass desired filename if you don't
        want to use default filename
    uniform_file: str
        full path of output uniform file
    max_file: str
        full path of output max file
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
    """
        
    chl_img = imio.imread(chl_file)
    chl_img = np.nan_to_num(chl_img)
    
    chl_img = np.ma.masked_less_equal(chl_img, bkg_thres)
    chl_img[chl_img.mask] = 0
    chl_img = np.uint16(chl_img*1000)
    
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
    log_file = log_file.replace('tif', 'txt')
    
    selem = skimage.morphology.disk(selem_size)
    loc_mean = skimage.filters.rank.mean(chl_img, selem=selem)
    loc_mean = np.float32(loc_mean)/1000.0
    rastertools.write_geotiff_with_source(chl_file, loc_mean, loc_mean_file,
                                          nodata=0, compress=False)
    
    #perc_mean = skimage.filters.rank.mean_percentile(chl_img, selem=selem,p0=.1, p1=.9)
    #perc_mean = np.float32(perc_mean)/1000.0
    #rastertools.write_geotiff_with_source(chl_file, perc_mean, perc_mean_file, nodata=0, compress=False)
    
    if labels == False:
        labels = None
        labelsYN = False
    else:
        labelsYN = True
        bw = skimage.filters.threshold_adaptive(loc_mean, 11)
        distance = scipy.ndimage.distance_transform_edt(bw)
        loc_max = skimage.feature.peak_local_max(distance, indices=False,
                                                 min_distance=min_distance,
                                                 labels=bw)
        markers = scipy.ndimage.label(loc_max)[0]
        labels = skimage.morphology.watershed(-distance, markers, mask=bw)
        label_file = loc_mean_file.replace('loc_mean', 'label')
        
        if small_thres is None:
            labels = labels
        else:
            small_selem = np.ones((small_thres, small_thres))
            small_obj = skimage.morphology.binary_opening(labels,
                                                          selem=small_selem)
            labels[~small_obj]=0  

        rastertools.write_geotiff_with_source(chl_file, labels, label_file,
                                              nodata=0, compress=False)    
    
    if labels is None:
        chl_img_max = skimage.feature.peak_local_max(loc_mean,
                                                     min_distance=min_distance,                                                 
                                                     exclude_border=True,
                                                     indices=False,
                                                     labels=labels)
    else:
        chl_img_max = skimage.feature.peak_local_max(loc_mean,
                                                     min_distance=min_distance,                                                 
                                                     exclude_border=False,
                                                     indices=False,
                                                     labels=labels)
                                                     
    uniform = classify.uniform_trees(loc_mean, chl_img_max, radius=radius)
    rastertools.write_geotiff_with_source(chl_file, uniform, uniform_file,
                                          nodata=0, compress=False)
    loc_mean[~chl_img_max] = float('NaN')
    rastertools.write_geotiff_with_source(chl_file, loc_mean, max_file,
                                          nodata=0, compress=False)
    print 'max', np.max(loc_mean[chl_img_max])
    print 'mean', np.mean(loc_mean[chl_img_max])
    print 'std dev', np.std(loc_mean[chl_img_max])
    print
    print np.mean(loc_mean[chl_img_max])*.5,np.mean(loc_mean[chl_img_max])*.8
    print np.mean(loc_mean[chl_img_max])*.8,np.mean(loc_mean[chl_img_max])*.9
    print np.mean(loc_mean[chl_img_max])*.9,np.mean(loc_mean[chl_img_max])*1.1
    print np.mean(loc_mean[chl_img_max])*1.1,np.mean(loc_mean[chl_img_max])*1.2
    print np.mean(loc_mean[chl_img_max])*1.2,np.mean(loc_mean[chl_img_max])*1.5      
    
    log = open(log_file, 'w')
    log.write(" background threshold=%s\n" %(bkg_thres))    
    log.write(" selem_size=%s\n min_distance=%s\n" %(selem_size,min_distance)) 
    log.write(" radius=%s\n" %(radius))
    log.write(" labels=%s\n small_thres=%s\n" %(labelsYN, small_thres))
    log.write(" max_value=%s\n" %(np.max(loc_mean[chl_img_max])))
    log.write(" mean_value=%s\n" %(np.mean(loc_mean[chl_img_max])))
    log.write(" std_dev=%s\n" %(np.std(loc_mean[chl_img_max])))
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
        
 
def chl_classi_loc_max(chl_file, ndvi=False, uniform_file=None, max_file=None,
                       radius=7, min_distance=3, labels=False, small_thres=None):
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
    
    #ndvi_img = improc.imops.imio.imread(ndvi_files[1])
    chl_img = imio.imread(chl_file)
    #ndvi_img = np.nan_to_num(ndvi_img)
    chl_img = np.nan_to_num(chl_img)
    #ndvi_img = improc.tests.anisodiff2y3d.anisodiff(ndvi_img, niter=3, kappa=80, gamma=0.2)
    tmp_img = anisodiff2y3d.anisodiff(chl_img, niter=3, kappa=80, gamma=0.2)
    
    if labels == False:
        labels = None
        labelsYN = False
    else:
        labelsYN = True
        bw = skimage.filters.threshold_adaptive(tmp_img, 11)
        distance = scipy.ndimage.distance_transform_edt(bw)
        loc_max = skimage.feature.peak_local_max(distance, indices=False,
                                                 min_distance=min_distance,
                                                 labels=bw)
        markers = scipy.ndimage.label(loc_max)[0]
        labels = skimage.morphology.watershed(-distance, markers, mask=bw)
        label_file = chl_file.replace('output', 'color/loc_max')
        if ndvi:
            label_file = label_file.replace('NDVI', 'NDVI_label')
        else:
            label_file = label_file.replace('chl', 'chl_label')
        rastertools.write_geotiff_with_source(chl_file, labels, label_file,
                                              nodata=0, compress=False)
        
    if small_thres is None:
        labels = labels
    else:
        small_selem = np.ones((small_thres, small_thres))
        small_obj = skimage.morphology.binary_opening(labels, selem=small_selem)
        labels[~small_obj] = 0
        
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
    
    log_file = max_file.replace('max', 'log')
    log_file = log_file.replace('tif', 'txt')
    
    if labels is None:
        chl_img_max = skimage.feature.peak_local_max(loc_mean,
                                                     min_distance=min_distance,                                                 
                                                     exclude_border=True,
                                                     indices=False,
                                                     labels=labels)
    else:
        chl_img_max = skimage.feature.peak_local_max(loc_mean,
                                                     min_distance=1,                                                 
                                                     exclude_border=False,
                                                     indices=False,
                                                     labels=labels)
                                                     
    uniform = classify.uniform_trees(chl_img, chl_img_max, radius=radius)
        
    print 'max', np.max(chl_img[chl_img_max])
    print 'mean', np.mean(chl_img[chl_img_max])
    print 'std dev', np.std(chl_img[chl_img_max])
    print
    print np.mean(chl_img[chl_img_max])*.5,np.mean(chl_img[chl_img_max])*.8
    print np.mean(chl_img[chl_img_max])*.8,np.mean(chl_img[chl_img_max])*.9
    print np.mean(chl_img[chl_img_max])*.9,np.mean(chl_img[chl_img_max])*1.1
    print np.mean(chl_img[chl_img_max])*1.1,np.mean(chl_img[chl_img_max])*1.2
    print np.mean(chl_img[chl_img_max])*1.2,np.mean(chl_img[chl_img_max])*1.5
    
    log = open(log_file, 'w')
    log.write(" min_distance=%s\n radius=%s\n" %(min_distance, radius))
    log.write(" labels=%s\n small_thres=%s\n" %(labelsYN, small_thres))
    log.write(" max_value=%s\n" %(np.max(chl_img[chl_img_max])))
    log.write(" mean_value=%s\n" %(np.mean(chl_img[chl_img_max])))
    log.write(" std_dev=%s\n" %(np.std(chl_img[chl_img_max])))
    log.close() 
    
    rastertools.write_geotiff_with_source(chl_file, uniform, uniform_file,
                                          nodata=-1, compress=False)
    chl_img[~chl_img_max] = float('NaN')
    rastertools.write_geotiff_with_source(chl_file, chl_img, max_file,
                                          nodata=-1, compress=False)
                                          

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
    
    
def chl_classi_loc_mean_v2(chl_file, bkg_thres=0.4, ndvi=False,
                           loc_mean_file=None, uniform_file=None,
                           max_file=None, selem_size=4, min_distance=3,
                           radius=7, labels=False, small_thres=None,
                           quicklook=False):
    """
    sandbox area for chl image classification
    this version has skimage local mean feature in it
    this version manually find local max of local mean
    
    Parameters
    ----------
    chl_file: str
        full path of input chl file
    bkg_thres: float
        threshold for masking out soil background and other stuff
    loc_mean_file: str
        full path of output local mean file. Pass desired filename if you don't
        want to use default filename
    uniform_file: str
        full path of output uniform file
    max_file: str
        full path of output max file
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
    """
        
    chl_img = imio.imread(chl_file)
    chl_img = np.nan_to_num(chl_img)
    
    chl_img = np.ma.masked_less_equal(chl_img, bkg_thres)
    chl_img[chl_img.mask] = 0
    chl_img = np.uint16(chl_img*1000)
    
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
    log_file = log_file.replace('tif', 'txt')
    
    selem = skimage.morphology.disk(selem_size)
    loc_mean = skimage.filters.rank.mean(chl_img, selem=selem)
    loc_mean = np.float32(loc_mean)/1000.0
    rastertools.write_geotiff_with_source(chl_file, loc_mean, loc_mean_file,
                                          nodata=0, compress=False)
    
    #perc_mean = skimage.filters.rank.mean_percentile(chl_img, selem=selem,p0=.1, p1=.9)
    #perc_mean = np.float32(perc_mean)/1000.0
    #rastertools.write_geotiff_with_source(chl_file, perc_mean, perc_mean_file, nodata=0, compress=False)
    
    if labels == False:
        labels = None
        labelsYN = False
    else:
        labelsYN = True
        bw = skimage.filters.threshold_adaptive(loc_mean, 11)
        distance = scipy.ndimage.distance_transform_edt(bw)
        loc_max = skimage.feature.peak_local_max(distance, indices=False,
                                                 min_distance=min_distance,
                                                 labels=bw)
        markers = scipy.ndimage.label(loc_max)[0]
        labels = skimage.morphology.watershed(-distance, markers, mask=bw)
        label_file = loc_mean_file.replace('loc_mean', 'label')
        
        if small_thres is None:
            labels = labels
        else:
            small_selem = np.ones((small_thres, small_thres))
            small_obj = skimage.morphology.binary_opening(labels,
                                                          selem=small_selem)
            labels[~small_obj]=0  

        rastertools.write_geotiff_with_source(chl_file, labels, label_file,
                                              nodata=0, compress=False)    
    
    if labels is None:
        chl_img_max = skimage.feature.peak_local_max(loc_mean,
                                                     min_distance=min_distance,                                                 
                                                     exclude_border=True,
                                                     indices=False,
                                                     labels=labels)
    else:
        # this sectoin manaully finds local max
        statCC = skimage.measure.regionprops(labels)
        uniform = np.zeros(chl_img.shape, dtype=loc_mean.dtype)
        chl_img_max = np.zeros(chl_img.shape, dtype=int)
        #data = np.empty(len(statCC), dtype=loc_mean.dtype)
    
        for i in range(len(statCC)):
            bb = statCC[i].bbox
            x1 = bb[0]; y1 = bb[1]; x2 = bb[2]; y2 = bb[3]
            uniform[x1:x2, y1:y2] = np.max(loc_mean[x1:x2, y1:y2])
            
            for j in range(x1, x2+1):   # this part is kind of stupid
                for k in range(y1, y2+1):
                    if loc_mean[j,k] == np.max(loc_mean[x1:x2, y1:y2]):
                        chl_img_max[j,k] = 1
            
        chl_img_max = chl_img_max.astype('bool')
        loc_mean[~chl_img_max] = float('NaN')
        rastertools.write_geotiff_with_source(chl_file, loc_mean, max_file,
                                                  nodata=0, compress=False)
        rastertools.write_geotiff_with_source(chl_file, uniform, uniform_file,
                                                  nodata=0, compress=False)
    
    print 'max', np.max(loc_mean[chl_img_max])
    print 'mean', np.mean(loc_mean[chl_img_max])
    print 'std dev', np.std(loc_mean[chl_img_max])
    print
    print np.mean(loc_mean[chl_img_max])*.5,np.mean(loc_mean[chl_img_max])*.8
    print np.mean(loc_mean[chl_img_max])*.8,np.mean(loc_mean[chl_img_max])*.9
    print np.mean(loc_mean[chl_img_max])*.9,np.mean(loc_mean[chl_img_max])*1.1
    print np.mean(loc_mean[chl_img_max])*1.1,np.mean(loc_mean[chl_img_max])*1.2
    print np.mean(loc_mean[chl_img_max])*1.2,np.mean(loc_mean[chl_img_max])*1.5      
    
    log = open(log_file, 'w')
    log.write(" chl_classi_loc_mean_v2")    
    log.write(" background threshold=%s\n" %(bkg_thres))    
    log.write(" selem_size=%s\n min_distance=%s\n" %(selem_size,min_distance)) 
    log.write(" radius=%s\n" %(radius))
    log.write(" labels=%s\n small_thres=%s\n" %(labelsYN, small_thres))
    log.write(" max_value=%s\n" %(np.max(loc_mean[chl_img_max])))
    log.write(" mean_value=%s\n" %(np.mean(loc_mean[chl_img_max])))
    log.write(" std_dev=%s\n" %(np.std(loc_mean[chl_img_max])))
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
        
