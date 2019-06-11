# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import glob

def proc_sen2cor(in_dir, f_pattern = '*MSIL1C*SAFE'):
    """
    calls sen2cor to perform atm cor on Sentinel 2A/B L1C data
    
    Parameters:
    ----
    in_dir: str
        the folder where stitched photos are
    """ 
    
    
    f_list = glob.glob(in_dir + f_pattern)
    
    for f in f_list:
        cmd='L2A_Process ' + f
        #cmd='L2A_Process ' + f + ' --resolution=10'
        #cmd='L2A_Process D:\sen_test\S2A_MSIL1C_20170101T182742_N0204_R127_T11SNS_20170101T182743.SAFE --resolution=10'
        os.system(cmd)   
