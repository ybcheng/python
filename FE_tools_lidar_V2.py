# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 11:23:05 2018

UPDATE:
implementaion of laspy to improve processing speed

@author: ybcheng
"""


import sys
import os
import glob
#import exiftool
import base64
import struct
import shutil
import copy
import math
import time
#import gdal
import laspy


#import improc

import pandas as pd
import numpy as np


def OpenLasFile(Filename):
    InputFile = laspy.file.File(Filename, mode='r')
    Data = np.vstack([InputFile.x, InputFile.y, InputFile.z, InputFile.intensity.astype(float),
                      InputFile.red.astype(float), InputFile.green.astype(float), InputFile.blue.astype(float),  
                      InputFile.num_returns.astype(float), InputFile.return_num.astype(float),
                      InputFile.Raw_Classification.astype(int)]).transpose()
    DF = pd.DataFrame(Data, columns = ['X', 'Y', 'Z', 'Intensity', 
                                       'Red', 'Green', 'Blue',
                                       'NumberOfReturns', 'ReturnNumber', 'Classification'])
    Header = InputFile.header
    InputFile.close()
    return DF, Header
    
    
def proc_veg_height(in_file, grid_size, mean_file=None, std_file=None, grnd_cls=2, veg_cls=4):
    """
    tool to calculate average veg height and std dev 
        -- requires well classified point cloud and transformation to dz
    """
    
    start_time = time.time()
    
    #if in_file.endswith('.csv'):
    #    print()
    #elif in_file.endswith('.txt'):
    #    os.rename(in_file, in_file.replace('.txt','.csv'))        
    #    in_file = in_file.replace('.txt','.csv')
    #else:        
    #    print('ERROR: input .csv file.')
    #    return      
    
    print(in_file)
    
    #in_df = pd.read_csv(in_file, sep=None, header=None, engine='python')
    #in_df.columns = colnames #['c', 'x' , 'y', 'z']
    
    in_df, header = OpenLasFile(in_file)
    in_df = in_df.loc[in_df['Z'] >= 0.0]    
    
    #in_df['veg'] = in_df.iloc[:,cls_col].apply(lambda x: x==veg_cls)
    #in_df['veg'] = in_df['veg'].astype(int)
    
    init_x = math.ceil(np.min(in_df.X))
    init_y = math.ceil(np.min(in_df.Y))
    
    dim_x = math.floor((np.max(in_df.X)-init_x)/grid_size) + 1
    dim_y = math.floor((np.max(in_df.Y)-init_y)/grid_size) + 1
    
    end_x = init_x + (dim_x - 1) * grid_size
    end_y = init_y + (dim_y - 1) * grid_size
    
    out_df = pd.DataFrame(index=np.arange(dim_x*dim_y),
                          columns=['x', 'y', 'mean', 'stdev'], dtype=float)
                          
    out_df.x = list(np.arange(init_x, end_x+grid_size, grid_size)) * dim_y
    out_df.y = sorted(list(np.arange(init_y, end_y+grid_size, grid_size)) * dim_x)
        
   
    for i in np.arange(out_df.shape[0]):
        if ((i%1000) == 0):
            print("%i / %i --- %.2f seconds" % (i, out_df.shape[0], (time.time()-start_time)))
                    
        up_x = out_df.x[i] + 0.5*grid_size
        lo_x = out_df.x[i] - 0.5*grid_size
        up_y = out_df.y[i] + 0.5*grid_size
        lo_y = out_df.y[i] - 0.5*grid_size
        
        tmp_df = in_df.loc[(in_df.X > lo_x) & (in_df.X < up_x) &
                           (in_df.Y > lo_y) & (in_df.Y < up_y)]
        
        if (len(tmp_df) == 0):
            out_df.loc[i,'mean'] = 0
            out_df.loc[i,'stdev'] = 0
        else:
            out_df.loc[i, 'mean']  = np.average(tmp_df.Z)
            out_df.loc[i, 'stdev'] = np.std(tmp_df.Z)                  
    
    mean_df = out_df[['x','y','mean' ]]
    std_df  = out_df[['x','y','stdev']]    
        
    
    if mean_file is None:
        mean_file = in_file.replace('.las', '_mean.csv')
    if std_file is None:
        std_file = in_file.replace('.las', '_std.csv')
        
    mean_df.to_csv(mean_file, index=False, sep=',')
    std_df.to_csv(std_file, index=False, sep=',')
    
    print("Done in --- %.2f seconds ---" % (time.time() - start_time))
    
    return mean_file, std_file
















def proc_dpf_las (in_file, ct_file, veg_cls=4, grnd_cls=2, keep_default=True):
    """
    reads in CENZ in csv format, merge with DPF centroids information
    then calculate %cover for each DPF locations
    """
    
    start_time = time.time()
        
    if in_file.endswith('.csv'):
        print()
    elif in_file.endswith('.txt'):
        os.rename(in_file, in_file.replace('.txt','.csv'))        
        in_file = in_file.replace('.txt','.csv')
    else:
        print('ERROR: input .csv file.')
        return
    
    in_df = pd.read_csv(in_file, sep=None, header=None, engine='python')
    in_df.columns = ['c', 'x' , 'y', 'z']
    in_df['name'] = ""
    
    ct_df = pd.read_csv(ct_file, sep=None, engine='python')
        
    new_df = pd.DataFrame()    
    for i in np.arange(ct_df.shape[0]):
        lo_x = ct_df.UTM_X[i]-2.5
        up_x = ct_df.UTM_X[i]+2.5
        lo_y = ct_df.UTM_Y[i]-2.5
        up_y = ct_df.UTM_Y[i]+2.5
        
        temp_df = pd.DataFrame()        
        temp_df = in_df.loc[(in_df.x > lo_x) & (in_df.x < up_x) &
                            (in_df.y > lo_y) & (in_df.y < up_y)]

        if (temp_df.shape[0] > 0):
            temp_df.loc[:,'name'] =  ct_df.loc[i,'Name']
            new_df = new_df.append(temp_df, ignore_index=True)       
          
    for n in set(new_df.name):
        tmp_df = pd.DataFrame()
        tmp_df = new_df.loc[new_df['name'] == n]
        
        if keep_default is not True:   #optional step to get rid of default class, use only ground and veg class
            tmp_df = tmp_df.loc[tmp_df['c'] != 1]
        
        tmp_df['veg'] = tmp_df.iloc[:,0].apply(lambda x: x==veg_cls)
        tmp_df['veg'] = tmp_df['veg'].astype(int)
        print(n, np.average(tmp_df.veg))
        #print(n, np.max(tmp_df.z))        
    
    print("Done in --- %.2f seconds ---" % (time.time() - start_time))
    
    #return new_df
    
    
def proc_dpf_range (in_file, ct_file, lo_h, hi_h, veg_cls=4, grnd_cls=2,
                        keep_default=True):
    """
    reads in CENZ in csv format, merge with DPF centroids information
    then calculate %cover for each DPF locations
    """
    
    #start_time = time.time()
        
    if in_file.endswith('.csv'):
        print()
    else:
        print('ERROR: input .csv file.')
        return
    
    in_df = pd.read_csv(in_file, sep=None, header=None, engine='python')
    in_df.columns = ['c', 'x' , 'y', 'z']
    in_df['name'] = ""
    
    ct_df = pd.read_csv(ct_file, sep=None, engine='python')
        
    new_df = pd.DataFrame()    
    for i in np.arange(ct_df.shape[0]):
        lo_x = ct_df.UTM_X[i]-2.5
        up_x = ct_df.UTM_X[i]+2.5
        lo_y = ct_df.UTM_Y[i]-2.5
        up_y = ct_df.UTM_Y[i]+2.5
        
        temp_df = pd.DataFrame()        
        temp_df = in_df.loc[(in_df.x > lo_x) & (in_df.x < up_x) &
                            (in_df.y > lo_y) & (in_df.y < up_y)]

        if (temp_df.shape[0] > 0):
            temp_df.loc[:,'name'] =  ct_df.loc[i,'Name']
            new_df = new_df.append(temp_df, ignore_index=True)       
          
    for n in set(new_df.name):
        tmp_df = pd.DataFrame()
        tmp_df = new_df.loc[new_df['name'] == n]
        
        if keep_default is not True:   #optional step to get rid of default class, use only ground and veg class
            tmp_df = tmp_df.loc[tmp_df['c'] != 1]
        
        count_total = tmp_df.shape[0]
        tmp_veg_rng = tmp_df.loc[(tmp_df.z >= lo_h) & (tmp_df.z < hi_h) &
                                   (tmp_df.c == veg_cls)]
        count_veg_rng = tmp_veg_rng.shape[0]
        
        #tmp_df['veg'] = tmp_df.iloc[:,0].apply(lambda x: x==veg_cls)
        #tmp_df['veg'] = tmp_df['veg'].astype(int)

        print(n, count_veg_rng/count_total)        
    
    #print("Done in --- %.2f seconds ---" % (time.time() - start_time))
    
    #return new_df
    
 

def proc_lascanopy (in_file, grid_size, cut_off, out_file):
    """
    lascanopy like tool
    """
        
    start_time = time.time()
    
    in_df = pd.read_csv(in_file, sep=None, header=None, engine='python')
    in_df.columns = ['x' , 'y', 'z']
    
    in_df['veg'] = in_df.iloc[:,2].apply(lambda x: x>cut_off)
    in_df['veg'] = in_df['veg'].astype(int)
    
    init_x = math.ceil(np.min(in_df.x))
    init_y = math.ceil(np.min(in_df.y))
    
    dim_x = math.floor((np.max(in_df.x)-init_x)/grid_size) + 1
    dim_y = math.floor((np.max(in_df.y)-init_y)/grid_size) + 1
    
    end_x = init_x + (dim_x - 1) * grid_size
    end_y = init_y + (dim_y - 1) * grid_size
    
    out_df = pd.DataFrame(index=np.arange(dim_x*dim_y),
                          columns=['x', 'y','cov'], dtype=float)
                          
    out_df.x = list(np.arange(init_x, end_x+grid_size, grid_size)) * dim_y
    out_df.y = sorted(list(np.arange(init_y, end_y+grid_size, grid_size)) * dim_x)
   
    for i in np.arange(out_df.shape[0]):
        if ((i%1000) == 0):
            print("%i / %i --- %.2f seconds" % (i, out_df.shape[0], (time.time()-start_time)))
                    
        up_x = out_df.x[i] + 0.5*grid_size
        lo_x = out_df.x[i] - 0.5*grid_size
        up_y = out_df.y[i] + 0.5*grid_size
        lo_y = out_df.y[i] - 0.5*grid_size
        
        tmp_df = in_df.loc[(in_df.x > lo_x) & (in_df.x < up_x) &
                           (in_df.y > lo_y) & (in_df.y < up_y)]
        
        if (len(tmp_df) == 0):
            out_df.loc[i,'cov'] = -1
        else:
            out_df.loc[i, 'cov'] = np.average(tmp_df.veg)                   
    
    out_df.to_csv(out_file, index=False, sep=',')
    
    print("Done in --- %.2f seconds ---" % (time.time() - start_time))
    
    return out_file
    
    
def proc_cover_class(in_file, grid_size, out_file=None, colnames=['c', 'x', 'y','z'],
                      cls_col=0, grnd_cls=2, veg_cls=4):
    """
    lascanopy like tool -- but use class instead of cutoff
    """
    
    start_time = time.time()
    
    if in_file.endswith('.csv'):
        print()
    elif in_file.endswith('.txt'):
        os.rename(in_file, in_file.replace('.txt','.csv'))        
        in_file = in_file.replace('.txt','.csv')
    else:        
        print('ERROR: input .csv file.')
        return      
    
    in_df = pd.read_csv(in_file, sep=None, header=None, engine='python')
    in_df.columns = colnames #['c', 'x' , 'y', 'z']
      
    in_df['veg'] = in_df.iloc[:,cls_col].apply(lambda x: x==veg_cls)
    in_df['veg'] = in_df['veg'].astype(int)
    
    init_x = math.ceil(np.min(in_df.x))
    init_y = math.ceil(np.min(in_df.y))
    
    dim_x = math.floor((np.max(in_df.x)-init_x)/grid_size) + 1
    dim_y = math.floor((np.max(in_df.y)-init_y)/grid_size) + 1
    
    end_x = init_x + (dim_x - 1) * grid_size
    end_y = init_y + (dim_y - 1) * grid_size
    
    out_df = pd.DataFrame(index=np.arange(dim_x*dim_y),
                          columns=['x', 'y','cov'], dtype=float)
                          
    out_df.x = list(np.arange(init_x, end_x+grid_size, grid_size)) * dim_y
    out_df.y = sorted(list(np.arange(init_y, end_y+grid_size, grid_size)) * dim_x)
   
    for i in np.arange(out_df.shape[0]):
        if ((i%1000) == 0):
            print("%i / %i --- %.2f seconds" % (i, out_df.shape[0], (time.time()-start_time)))
                    
        up_x = out_df.x[i] + 0.5*grid_size
        lo_x = out_df.x[i] - 0.5*grid_size
        up_y = out_df.y[i] + 0.5*grid_size
        lo_y = out_df.y[i] - 0.5*grid_size
        
        tmp_df = in_df.loc[(in_df.x > lo_x) & (in_df.x < up_x) &
                           (in_df.y > lo_y) & (in_df.y < up_y)]
        
        if (len(tmp_df) == 0):
            out_df.loc[i,'cov'] = -1
        else:
            out_df.loc[i, 'cov'] = np.average(tmp_df.veg)                   
    
    if out_file is None:
        out_file = in_file.replace('.csv', '_cov.csv')
        
    out_df.to_csv(out_file, index=False, sep=',')
    
    print("Done in --- %.2f seconds ---" % (time.time() - start_time))
    
    return out_file


def proc_cover_range (in_file, grid_size, lo_h, hi_h, out_file=None, 
                          cls_col=0, grnd_cls=2, veg_cls=4,
                          colnames=['c','x','y','z']):

    """
    calculate percent vegetative cover within a height range
    """

    
    start_time = time.time()
    
    if in_file.endswith('.csv'):
        print()
    elif in_file.endswith('.txt'):
        os.rename(in_file, in_file.replace('.txt','.csv'))        
        in_file = in_file.replace('.txt','.csv')
    else:        
        print('ERROR: input .csv file.')
        return 
    
    in_df = pd.read_csv(in_file, sep=None, header=None, engine='python')
    in_df.columns = colnames #['c', 'x' , 'y', 'z']
    
    #in_df['veg'] = in_df.iloc[:,cls_col].apply(lambda x: x==veg_cls)
    #in_df['veg'] = in_df['veg'].astype(int)
    
    init_x = math.ceil(np.min(in_df.x))
    init_y = math.ceil(np.min(in_df.y))
    
    dim_x = math.floor((np.max(in_df.x)-init_x)/grid_size) + 1
    dim_y = math.floor((np.max(in_df.y)-init_y)/grid_size) + 1
    
    end_x = init_x + (dim_x - 1) * grid_size
    end_y = init_y + (dim_y - 1) * grid_size
    
    out_df = pd.DataFrame(index=np.arange(dim_x*dim_y),
                          columns=['x', 'y','cov'], dtype=float)
                          
    out_df.x = list(np.arange(init_x, end_x+grid_size, grid_size)) * dim_y
    out_df.y = sorted(list(np.arange(init_y, end_y+grid_size, grid_size)) * dim_x)
   
    for i in np.arange(out_df.shape[0]):
        if ((i%1000) == 0):
            print("%i / %i --- %.2f seconds" % (i, out_df.shape[0], (time.time()-start_time)))
                    
        up_x = out_df.x[i] + 0.5*grid_size
        lo_x = out_df.x[i] - 0.5*grid_size
        up_y = out_df.y[i] + 0.5*grid_size
        lo_y = out_df.y[i] - 0.5*grid_size
        
        tmp_df = in_df.loc[(in_df.x > lo_x) & (in_df.x < up_x) &
                           (in_df.y > lo_y) & (in_df.y < up_y)]
        count_total = tmp_df.shape[0]
        
        tmp_df_veg = tmp_df.loc[(tmp_df.c == veg_cls) & (tmp_df.z >= lo_h) & (tmp_df.z < hi_h)]
        count_veg = tmp_df_veg.shape[0]
        
        if (len(tmp_df) == 0):
            out_df.loc[i, 'cov'] = -1
        else:
            out_df.loc[i, 'cov'] = count_veg / count_total
    
    if out_file is None:
        append_fn = '_cov_' + str(lo_h) + '_' + str(hi_h) + '.csv'
        out_file = in_file.replace('.csv', append_fn)
        
    out_df.to_csv(out_file, index=False, sep=',')
    
    print("Done in --- %.2f seconds ---" % (time.time() - start_time))
        
    #return out_file
    
    
def proc_max_hght (in_file, grid_size, out_file=None, colnames=['c','x','y','z'],
                   veg_cls=4):                          #cls_col=0, grnd_cls=2, ,
                          
    """
    calculate percent vegetative cover within a height range
    """

    
    start_time = time.time()
    
    if in_file.endswith('.csv'):
        print()
    elif in_file.endswith('.txt'):
        os.rename(in_file, in_file.replace('.txt','.csv'))        
        in_file = in_file.replace('.txt','.csv')
    else:        
        print('ERROR: input .csv file.')
        return 
    
    in_df = pd.read_csv(in_file, sep=None, header=None, engine='python')
    in_df.columns = colnames #['c', 'x' , 'y', 'z']
    
    #in_df['veg'] = in_df.iloc[:,cls_col].apply(lambda x: x==veg_cls)
    #in_df['veg'] = in_df['veg'].astype(int)
    
    init_x = math.ceil(np.min(in_df.x))
    init_y = math.ceil(np.min(in_df.y))
    
    dim_x = math.floor((np.max(in_df.x)-init_x)/grid_size) + 1
    dim_y = math.floor((np.max(in_df.y)-init_y)/grid_size) + 1
    
    end_x = init_x + (dim_x - 1) * grid_size
    end_y = init_y + (dim_y - 1) * grid_size
    
    out_df = pd.DataFrame(index=np.arange(dim_x*dim_y),
                          columns=['x', 'y','max'], dtype=float)
                          
    out_df.x = list(np.arange(init_x, end_x+grid_size, grid_size)) * dim_y
    out_df.y = sorted(list(np.arange(init_y, end_y+grid_size, grid_size)) * dim_x)
   
    for i in np.arange(out_df.shape[0]):
        if ((i%1000) == 0):
            print("%i / %i --- %.2f seconds" % (i, out_df.shape[0], (time.time()-start_time)))
                    
        up_x = out_df.x[i] + 0.5*grid_size
        lo_x = out_df.x[i] - 0.5*grid_size
        up_y = out_df.y[i] + 0.5*grid_size
        lo_y = out_df.y[i] - 0.5*grid_size
        
        tmp_df = in_df.loc[(in_df.x > lo_x) & (in_df.x < up_x) &
                           (in_df.y > lo_y) & (in_df.y < up_y)]
                           
        tmp_df = tmp_df.loc[(tmp_df.c == veg_cls)]
        
        if (len(tmp_df) == 0):
            out_df.loc[i, 'max'] = 0
        else:
            out_df.loc[i, 'max'] = np.max(tmp_df.z)
    
    if out_file is None:
        out_file = in_file.replace('.csv', '_max.csv')
        
    out_df.to_csv(out_file, index=False, sep=',')
    
    print("Done in --- %.2f seconds ---" % (time.time() - start_time))

    
def proc_IQR_hght (in_file, grid_size, out_file=None, colnames=['c','x','y','z'],
                   veg_cls=4):                          #cls_col=0, grnd_cls=2, ,
                          
    """
    calculate percent vegetative cover within a height range
    """

    
    start_time = time.time()
    
    if in_file.endswith('.csv'):
        print()
    elif in_file.endswith('.txt'):
        os.rename(in_file, in_file.replace('.txt','.csv'))        
        in_file = in_file.replace('.txt','.csv')
    else:        
        print('ERROR: input .csv file.')
        return 
    
    in_df = pd.read_csv(in_file, sep=None, header=None, engine='python')
    in_df.columns = colnames #['c', 'x' , 'y', 'z']
    
    #in_df['veg'] = in_df.iloc[:,cls_col].apply(lambda x: x==veg_cls)
    #in_df['veg'] = in_df['veg'].astype(int)
    
    init_x = math.ceil(np.min(in_df.x))
    init_y = math.ceil(np.min(in_df.y))
    
    dim_x = math.floor((np.max(in_df.x)-init_x)/grid_size) + 1
    dim_y = math.floor((np.max(in_df.y)-init_y)/grid_size) + 1
    
    end_x = init_x + (dim_x - 1) * grid_size
    end_y = init_y + (dim_y - 1) * grid_size
    
    out_df = pd.DataFrame(index=np.arange(dim_x*dim_y),
                          columns=['x', 'y','iqr'], dtype=float)
                          
    out_df.x = list(np.arange(init_x, end_x+grid_size, grid_size)) * dim_y
    out_df.y = sorted(list(np.arange(init_y, end_y+grid_size, grid_size)) * dim_x)
   
    for i in np.arange(out_df.shape[0]):
        if ((i%1000) == 0):
            print("%i / %i --- %.2f seconds" % (i, out_df.shape[0], (time.time()-start_time)))
                    
        up_x = out_df.x[i] + 0.5*grid_size
        lo_x = out_df.x[i] - 0.5*grid_size
        up_y = out_df.y[i] + 0.5*grid_size
        lo_y = out_df.y[i] - 0.5*grid_size
        
        tmp_df = in_df.loc[(in_df.x > lo_x) & (in_df.x < up_x) &
                           (in_df.y > lo_y) & (in_df.y < up_y)]
                           
        tmp_df = tmp_df.loc[(tmp_df.c == veg_cls)]
        
        if (len(tmp_df) == 0):
            out_df.loc[i, 'iqr'] = 0
        else:
            out_df.loc[i, 'iqr'] = np.percentile(tmp_df.z,75) - np.percentile(tmp_df.z,25)
    
    if out_file is None:
        out_file = in_file.replace('.csv', '_IQR.csv')
        
    out_df.to_csv(out_file, index=False, sep=',')
    
    print("Done in --- %.2f seconds ---" % (time.time() - start_time))
    
    
def proc_FHD (in_file, grid_size, out_file=None, colnames=['c','x','y','z'],
                   veg_cls=4, layer=[0,0.05,0.1,0.2,100]):                          #cls_col=0, grnd_cls=2, ,
                          
    """
    calculate percent vegetative cover within a height range
    """

    
    start_time = time.time()
    
    if in_file.endswith('.csv'):
        print()
    elif in_file.endswith('.txt'):
        os.rename(in_file, in_file.replace('.txt','.csv'))        
        in_file = in_file.replace('.txt','.csv')
    else:        
        print('ERROR: input .csv file.')
        return 
    
    in_df = pd.read_csv(in_file, sep=None, header=None, engine='python')
    in_df.columns = colnames #['c', 'x' , 'y', 'z']
    
    #in_df['veg'] = in_df.iloc[:,cls_col].apply(lambda x: x==veg_cls)
    #in_df['veg'] = in_df['veg'].astype(int)
    
    init_x = math.ceil(np.min(in_df.x))
    init_y = math.ceil(np.min(in_df.y))
    
    dim_x = math.floor((np.max(in_df.x)-init_x)/grid_size) + 1
    dim_y = math.floor((np.max(in_df.y)-init_y)/grid_size) + 1
    
    end_x = init_x + (dim_x - 1) * grid_size
    end_y = init_y + (dim_y - 1) * grid_size
    
    out_df = pd.DataFrame(index=np.arange(dim_x*dim_y),
                          columns=['x', 'y','fhd'], dtype=float)
                          
    out_df.x = list(np.arange(init_x, end_x+grid_size, grid_size)) * dim_y
    out_df.y = sorted(list(np.arange(init_y, end_y+grid_size, grid_size)) * dim_x)
   
    for i in np.arange(out_df.shape[0]):
        if ((i%1000) == 0):
            print("%i / %i --- %.2f seconds" % (i, out_df.shape[0], (time.time()-start_time)))
                    
        up_x = out_df.x[i] + 0.5*grid_size
        lo_x = out_df.x[i] - 0.5*grid_size
        up_y = out_df.y[i] + 0.5*grid_size
        lo_y = out_df.y[i] - 0.5*grid_size
        
        tmp_df = in_df.loc[(in_df.x > lo_x) & (in_df.x < up_x) &
                           (in_df.y > lo_y) & (in_df.y < up_y)]
                           
        tmp_df = tmp_df.loc[(tmp_df.c == veg_cls)]
        p = np.empty(len(layer)-1)
        
        if (len(tmp_df) == 0):
            out_df.loc[i, 'fhd'] = -1
        else:
            count_total = tmp_df.shape[0]
            
            for j in np.arange(len(p)):
                tmp_df2 = tmp_df.loc[(tmp_df.z >= layer[j]) & (tmp_df.z < layer[j+1])]
                p[j] = tmp_df2.shape[0] / count_total
                            
            p[p==0] = 1
            ln_p = [math.log(z) for z in p]            
            out_df.loc[i, 'fhd'] = -np.dot(p, ln_p)
    
    if out_file is None:
        out_file = in_file.replace('.csv', '_FHD.csv')
        
    out_df.to_csv(out_file, index=False, sep=',')
    
    print("Done in --- %.2f seconds ---" % (time.time() - start_time))
    
    
def csv2tif (source, target=None):
    """
    Could be handle little tool to convert csv results from proc_lascanopy
    to GeoTIFF
    
    NOTE: the output does NOT have correct GeoTag!!    
    
    https://gis.stackexchange.com/questions/177061/ascii-file-with-latitude-longitude-and-data-to-geotiff-using-python
    """
    
    
    if source.endswith('.csv'):
        print()
    else:
        print('ERROR: input .csv file')
        return
    
    cvs = gdal.Open(source)
    if cvs is None:
        print('ERROR: Unable to open %s' % source)
        return

    geotiff = gdal.GetDriverByName("GTiff")
    if geotiff is None:
        print('ERROR: GeoTIFF driver not available.')
        return

    if target is None:
        target = source.replace('.csv', '.tif')
    
    #options = ["EPSG=26911"]
    options = []    
    geotiff.CreateCopy(target, cvs, 0, options)
