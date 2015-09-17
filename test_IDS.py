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
import time
import copy
import fiona
import shapely
import matplotlib.pyplot as plt
from scipy import stats

from ..imops import imio
from ..gis import rastertools
from ..gen import strops
from ..cv import classify, slicing
from ..dbops import finder, loader


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
