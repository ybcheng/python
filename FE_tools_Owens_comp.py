# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:49:10 2017

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
import improc

from scipy import stats
from skimage import filters, morphology, feature

#from ..imops import imio, imcalcs
#from ..gis import rastertools, shapetools, extract
#from ..gen import strops, dirfuncs, wrappers
#from ..cv import classify, slicing
#from ..dbops import finder, loader, parse
#from . import anisodiff2y3d
#from improc.cv.genutils import square, disk


def sep_zonal_stat_files(file, out_dir, out_suffix='.csv'):
    """
    seperate zonal stats file for each DCA
    
    Parameters
    ----------
    file: string
        full path of zonal stats file
        e.g. J:\Owens_Lake\tasks\comp-mon\Veg\2016\Analysis\2016_Ph7a_compliance\ALL_1_stats.csv
    out_dir: string
        full path of where the output files should be stored
        e.g. J:\Owens_Lake\tasks\comp-mon\Veg\2016\Analysis\2016_Ph7a_compliance\
        see how zonal stats for each DCA were separated
    """
    
    DCAs = ['ChN', 'T1A-2','ChS','T28NS','T37-2','T32-1','T36-1w','T36-1e','T30-1','T5-1_Addition']
    zonal_DF = pd.read_csv(file)
    
    for d in DCAs:
        stats_DF = zonal_DF.loc[zonal_DF.StdPolygon == d]
        out_filename = out_dir + d + out_suffix
        stats_DF.to_csv(out_filename, index=False, sep=',')    


def gen_compliance_files_v2(stats_files, report_filename, replace=True):
    """
    simple wrapper for calc_compliance_v2
    """
    
    log = open(report_filename, 'w')
     
    for sf in stats_files:
        total_cells, stats_five, stats_ten, stats_twenty = calc_compliance_v2(sf)
        bfn = os.path.basename(sf)        
        log.write("%s, %s, %s, %s, %s\n" %(bfn,total_cells,stats_five,stats_ten,stats_twenty))
        
    log.close()
    print("done")


def calc_compliance_v2(file):
    """
    """
    
    stats_DF = pd.read_csv(file, sep=',')
    stats_DF = stats_DF.iloc[:,2:]
    stats_DF = stats_DF.dropna()
    
    total_cells = len(stats_DF['MEAN'])
    
    DF_five = stats_DF.loc[stats_DF.MEAN > 5]
    DF_ten = stats_DF.loc[stats_DF.MEAN > 10]
    DF_twenty = stats_DF.loc[stats_DF.MEAN > 20]
    
    count_five = len(DF_five['MEAN'])
    count_ten = len(DF_ten['MEAN'])
    count_twenty = len(DF_twenty['MEAN'])    
    
    stats_five = count_five / total_cells    
    stats_ten = count_ten / total_cells
    stats_twenty = count_twenty / total_cells
    
    
    #stats_DF['five'] = stats_DF['MEAN']>5
    #stats_DF['ten'] = stats_DF['MEAN']>10
    #stats_DF['twenty'] = stats_DF['MEAN']>20

    #count_five = stats_DF.groupby('five')['five'].count() 
    #count_ten = stats_DF.groupby('ten')['ten'].count()
    #count_twenty = stats_DF.groupby('twenty')['twenty'].count()
    
    #stats_five = count_five.loc[True] / len(stats_DF['five'])
    #stats_ten = count_ten.loc[True] / len(stats_DF['ten'])
    #stats_twenty = count_twenty.loc[True] / len(stats_DF['twenty'])
    
    return total_cells, stats_five, stats_ten, stats_twenty


def gen_compliance_files(stats_files, replace=True):
    """
    simple wrapper of calc_complinace and generates output_files
    
    Parameters
    ----------
    files: list
        list of full path of files to be processed
    """
    
    for sf in stats_files:
        if not ('stats.txt' in sf):
            print("check filename for %s" %(sf))
            continue        
        
        report_filename = sf.replace('stats.txt', 'report.csv')
        if os.path.exists(report_filename):
                if not replace:
                    print("%s already exists" %(report_filename))
                    continue
                else:
                    os.remove(report_filename)        
        
        try:
            avg_CE, avg_cover, std_cover, out_DF = calc_compliance(sf)
        except KeyError:
            print("Error generating report for %s" %(sf))
            continue
        
        log = open(report_filename, 'w')
        log.write("Average CE = %s\n" %(avg_CE))
        log.write("Average cover = %s\n" %(avg_cover))
        log.write("SD cover = %s\n" %(std_cover))
        log.close()
                        
        with open(report_filename, 'a') as rf:
            pd.DataFrame.to_csv(out_DF, rf)
            
        print("Generated report for %s" %(sf))


def calc_compliance(file):
    """
    calculate compliance report
    
    Parameters
    ----------
    file: str
        full path of output file from zonal stats
        the file needs to have 'MEAN' column as coverage
        and 'Shape_Area' as area of each polgon cell
    """
    
    stats_DF = pd.read_csv(file, sep=',')
    
    stats_DF['cover_class'] = np.floor_divide(stats_DF['MEAN'],5)*5 + 2.5
    stats_DF['CE'] = 100*(1-np.exp((-0.41865) * np.floor(stats_DF['MEAN'])))
    stats_DF['cover_area'] = stats_DF['MEAN'] * stats_DF['Shape_Area']
    
    avg_CE = np.nanmean(stats_DF['CE'])
    avg_cover = np.sum(stats_DF['cover_area']) / np.sum(stats_DF['Shape_Area'])
    std_cover = np.nanstd(stats_DF['MEAN'], ddof=1)
    
    o1 = stats_DF.groupby('cover_class')['cover_class'].count()
    o2 = stats_DF.groupby('cover_class')['Shape_Area'].sum()

    out_DF = pd.Series.to_frame(o1)
    out_DF.columns = ['Count of Class']
    out_DF['Sum of Area'] = o2
    out_DF.loc['Grand_Total'] = [np.sum(o1), np.sum(o2)]

    return avg_CE, avg_cover, std_cover, out_DF


def classi_loc_max(img_filepath, bg_thres, out_filepath):
    """
    adaptive threshold & watershed segmentation based procedure
    """
        
    img = improc.imops.imio.imread(img_filepath)
    img = np.ma.masked_less_equal(img, bg_thres)
    img[img.mask] = 0.0
    
    bw = skimage.filters.threshold_adaptive(img, 11)
    distance = scipy.ndimage.distance_transform_edt(bw)
    loc_max = skimage.feature.peak_local_max(distance, indices=False, 
                                             min_distance=13, labels=bw)
    markers = scipy.ndimage.label(loc_max)[0]
    labels = skimage.morphology.watershed(-distance, markers, mask=bw)
        
    improc.gis.rastertools.write_geotiff_with_source(img_filepath, labels,
                                                     out_filepath)
    
    
def distance_map(img_filepath, bg_thres, cov_filepath, seg_filepath=None,
                 bg_value=-1, radius=1, use_adaptive=True):
    """
    distance map based procedure
    """
    
    img = improc.imops.imio.imread(img_filepath)    
    #bg_thres = 0.38
    #bg_value = -1
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
        
    #seg_img = scipy.ndimage.grey_dilation(ndsi_img_mskd, footprint=exp_sq)
    seg_img = scipy.ndimage.grey_dilation(img, footprint=exp_dsk)
    cov_img = np.empty(seg_img.shape, 'uint8')
    seg_img = np.ma.masked_equal(seg_img, -1.0)
    cov_img[seg_img.mask] = 0
    cov_img[~seg_img.mask] = 1
    
    if seg_filepath is not None:
        improc.gis.rastertools.write_geotiff_with_source(img_filepath, seg_img,
                                                         seg_filepath)
    improc.gis.rastertools.write_geotiff_with_source(img_filepath, cov_img,
                                                     cov_filepath)