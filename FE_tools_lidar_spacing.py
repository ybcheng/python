# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 08:08:51 2018

@author: ybcheng
"""

from laspy.file import File
import numpy as np
import pandas as pd
#from sklearn.cluster import DBSCAN
#import collections
#import matplotlib.pyplot as plt
#import seaborn as sns
#import matplotlib
import time
import os
#sns.set(font_scale=3)
from scipy.spatial.distance import cdist
#sns.palplot(sns.color_palette("coolwarm", 7))


def proc_las_spacing (in_las, out_las=None):
    """
    reads in las 1.2 file format
    grab vegetation points
    then calculate spacing from wire for each vegetation point
    """
    
    start_time = time.time()

        
    if out_las is None:
        out_las = in_las.replace('.las', '_spacing.las')
        
    if os.path.exists(out_las):
        print('ERROR: output file already exists')
        return        
  
    
    File_ = File(in_las, mode='r')
    Data = np.vstack([File_.x, File_.y, File_.z, File_.num_returns, File_.return_num, File_.Raw_Classification,
                      File_.Red, File_.Green, File_.Blue, File_.Intensity]).transpose()
    DF = pd.DataFrame(Data, columns = ['X', 'Y', 'Z', 'NumberOfReturns', 'ReturnNumber', 'Classification',
                                       'Red', 'Green', 'Blue','Intensity'])


    Powerlines = DF[DF['Classification'] == 8]
    Tree = DF[DF['Classification'] == 4]

    dist = cdist(Powerlines[['X', 'Y', 'Z']].values, Tree[['X', 'Y', 'Z']].values, metric="euclidean")

    MinDistances = np.min(dist, axis = 0)

    Tree['MinDistances'] = MinDistances
    
    
        
    Outfile = File(out_las, mode = 'w', header=File_.header)

    MinDistances = 1000*MinDistances
    #MinDistances = 3.28084*MinDistances    
    Outfile.pt_src_id = np.asarray(MinDistances).astype(float)
    Outfile.x = Tree['X'].values
    Outfile.y = Tree['Y'].values
    Outfile.z = Tree['Z'].values
    Outfile.num_returns = Tree['NumberOfReturns'].values.astype(int)
    Outfile.return_num = Tree['ReturnNumber'].values.astype(int)
    Outfile.Raw_Classification = Tree['Classification'].values
    Outfile.Red = Tree['Red'].values
    Outfile.Green = Tree['Green'].values
    Outfile.Blue = Tree['Blue'].values
    Outfile.Intensity = Tree['Intensity'].values

    Outfile.close()

    print("Done in --- %.2f seconds ---" % (time.time() - start_time))