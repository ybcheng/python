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


def sep_zonal_stat_files(files):
    """
    simple warpper for sep_zonal_file
    """
    
    for f in files:
        print('processing: ' + f)        
        sep_zonal_stat_file(f)

    print('done')
    

def sep_zonal_stat_file(file, out_dir=None, out_suffix=None):
    """
    seperate zonal stats file for each DCA
    
    Parameters
    ----------
    file: string
        full path of zonal stats file
        e.g. J:/Owens_Lake/tasks/comp-mon/Veg/2017/analysis/20170623_ph7a_compliance/ALL_1_stats.csv
    out_dir: string
        full path of where the output files should be stored
        DEFAULT to the same directory as in the input files
        e.g. J:/Owens_Lake/tasks/comp-mon/Veg/2017/analysis/20170623_ph7a_compliance/
        see how zonal stats for each DCA were separated
        
    EXAMPLE use something like below in the console:
    files = glob.glob('J:/Owens_Lake/tasks/comp-mon/Veg/2017/analysis/20170623_ph7a_compliance/temp/ALL*csv')
    for f in files:
        FE_tools_Owens_comp.sep_zonal_stat_files(f)
    """
    
    DCAs = ['ChN', 'T01A-2','ChS','T28','T37-2','T32-1','T36-1W','T36-1E','T30-1','T5-1_Addition']
    zonal_DF = pd.read_csv(file)
    
    if out_dir is None:
        out_dir = os.path.dirname(file) + '/'
        
    for d in DCAs:
        stats_DF = zonal_DF.loc[zonal_DF.StdPolygon == d]
        if not stats_DF.empty:
            if out_suffix is None:
                out_filename = out_dir + os.path.basename(file).replace('ALL', d)
            else:
                out_filename = out_dir + d + out_suffix
            stats_DF.to_csv(out_filename, index=False, sep=',')    


def gen_compliance_files_v2(stats_files, report_filename=None, replace=True):
    """
    simple wrapper for calc_compliance_v2
    """
    
    if report_filename is None:
        report_filename = os.path.dirname(stats_files[0]) + '/report.csv'
        
    if os.path.exists(report_filename):
        if replace:
           os.remove(report_filename)
        else:           
           print("%s already exists, quitting..." %(report_filename))
           return
    
    log = open(report_filename, 'w')
     
    for sf in stats_files:
        total_cells, stats_five, stats_ten, stats_twenty = calc_compliance_v2(sf)
        bfn = os.path.basename(sf)        
        log.write("%s, %s, %s, %s, %s\n" %(bfn,total_cells,stats_five,stats_ten,stats_twenty))
        
    log.close()
    print("done")


def calc_compliance_v2(file):
    """
    updated function to calculate compliance for Owen's MV areas
    based on areas over 5%, 10%, 20% coverage
    
    Parameters
    ----------
    file: str
        full path of output file from zonal stats
        the file needs to have 'MEAN' column as coverage
        e.g. J:\Owens_Lake\tasks\comp-mon\Veg\2016\Analysis\2016_Ph7a_compliance\ChS_1_stats.csv
    """
    
    stats_DF = pd.read_csv(file, sep=',')
    #stats_DF = stats_DF.iloc[:,2:]  #dropping first column since it's empty, check input files for compatibility
    stats_DF['MEAN'] = pd.to_numeric(stats_DF['MEAN'], errors='coerce')    
    stats_DF = stats_DF.dropna()
    
    total_cells = len(stats_DF['MEAN'])
    
    DF_five = stats_DF.loc[stats_DF.MEAN > 0.05]
    DF_ten = stats_DF.loc[stats_DF.MEAN > 0.10]
    DF_twenty = stats_DF.loc[stats_DF.MEAN > 0.20]
    
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

