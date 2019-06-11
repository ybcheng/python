# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:42:08 2016
Utilities for processing Salton Sea soil core photos
@author: ybcheng
"""


import os
import sys
import glob
import fnmatch
from shutil import copyfile

import improc
import numpy as np


def ren_mosaic(mosaic_dir='K:/IID_SaltonSea/Tasks/Soil mapping/PhotoDocumentation/Original/', 
               file_pattern='*stitch.jpg'):
    """
    may be a handy little tool that renames mosaic filename
    so it matches folder name (core name)
    
    Parameters:
    ----
    mosaic_dir: str
        the folder where stitched photos are
    file_pattern: str
        stitched photos, if out of MS ICE, it should have _stitch appended
    """ 
        
   
    if not os.path.exists(mosaic_dir):
        sys.exit('input folder does not exist')
   
    mosaics = []
    for root, dirnames, filenames in os.walk(mosaic_dir):
        for filename in fnmatch.filter(filenames, file_pattern):
            mosaics.append(os.path.join(root, filename).replace('\\','/'))
            
    s = 0
    r = 0
    for m in mosaics:
        dir_name = os.path.dirname(m).split('/')[-1]
        new_name = os.path.dirname(m) + '/' + dir_name + '.jpg'
        if os.path.exists(new_name):
            print('skipping: %s' % m)
            s+=1
        else:
            os.rename(m, new_name)
            print('renamed: %s' % new_name)
            r+=1
            
    print('renamed total of %i files' % r)
    print('skipped total of %i files' % s)    


def copy_mosaic(mosaic_dir='K:/IID_SaltonSea/Tasks/Soil mapping/PhotoDocumentation/Original/',
                output_dir='K:/IID_SaltonSea/Tasks/Soil mapping/PhotoDocumentation/Processing/',
                file_pattern='IID201905*jpg', replace=False):
    """
    may be a handy little tool that copies soil core mosaic to "final" folder
    the program searches all the stitched photos in a dir and sub-dir
    and copies them to the "final" folder if they are not there already
    
    Parameters:
    ----
    mosaic_dir: str
        the folder where stitched photos are
    output_dir: str
        where the stitched photos should go
    file_pattern: str
        stitched photos should have the same name as soil cores
    """ 
        
    if not os.path.exists(mosaic_dir):
        sys.exit('input folder does not exist')
    
    mosaics = []
    for root, dirnames, filenames in os.walk(mosaic_dir):
        for filename in fnmatch.filter(filenames, file_pattern):
            mosaics.append(os.path.join(root, filename))
            
    c = 0
    s = 0
    r = 0
    for m in mosaics:
        f = output_dir + os.path.basename(m)
        if not os.path.exists(f):
            copyfile(m, f)
            print('copied: %s' % f)
            c+=1
        elif replace:
            copyfile(m, f)
            print('replaced: %s' % f)
            r+=1
        else:
            print('skipped: %s' % f)
            s+=1
            
    print('copied total of %i files' % c)
    print('replaced total of %i files' % r)
    print('skipped total of %i files' % s)


def rot_mosaic(source_dir='K:/IID_SaltonSea/Tasks/Soil mapping/PhotoDocumentation/Processing/',
               output_dir='K:/IID_SaltonSea/Tasks/Soil mapping/PhotoDocumentation/Final/',
               file_pattern='IID201905*.jpg', sub_dir=False, k=1, replace=False):
    """
    may be a handy little tool that rotates soil core mosaic in "final" folder
    the program searches all the stitched photos in a dir (including sub-dir)
    and rotates them
    
    Parameters:
    ----
    source_dir: str
        the folder where stitched photos are
    output_dir: str
        where the stitched photos should go
    file_pattern: str
        stitched photos should have the same name as soil cores
    sub_dir: bool
        search sub-directory or not, default to False
    k: int
        Number of times the array is rotated counter-clockwise 90 degrees
    replace: bool
        replace existing mosaic or not, default to False 
    """ 
        
    
    if sub_dir:
        mosaics = []
        for root, dirnames, filenames in os.walk(source_dir):
            for filename in fnmatch.filter(filenames, file_pattern):
                mosaics.append(os.path.join(root, filename))
    else:
         mosaics = glob.glob(source_dir + file_pattern)       
            
    g = 0
    r = 0
    s = 0
    for m in mosaics:
        f = output_dir + os.path.basename(m)
        if not os.path.exists(f):
            img = improc.imops.imio.imread(m)
            img = np.rot90(img, k=k)            
            improc.imops.imio.imsave(f, img)
            print('generated: %s' % f)
            print('')
            g+=1
        elif replace:
            img = improc.imops.imio.imread(m)
            img = np.rot90(img, k=k)
            improc.imops.imio.imsave(f, img)
            print('replaced: %s' % f)
            print('')
            r+=1
        else:
            print('skipping: %s' % m)
            print('')
            s+=1

    print('generated total of %i files' % g)
    print('replaced total of %i files' % r)
    print('skipped total of %i files' % s)           
            
            