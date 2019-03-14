# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:58:09 2018

@author: ybcheng

Source:
https://gis.stackexchange.com/questions/88800/removing-duplicates-from-large-las-files/88804
"""


import numpy as np
from laspy.file import File


def rmv_duplicate(in_laspath, out_laspath):
    """
    function to remove duplicate points from a las file. It's design to remove
    points at the "exact same location", works like:
    lasduplicate -unique_xyz
    
    Source:
    https://gis.stackexchange.com/questions/88800/removing-duplicates-from-large-las-files/88804
    
    Parameters:
    ----
    in_laspath: str
        input las filepath
    out_laspath: str
        output las filepath
    """
    
    
    inFile = File(in_laspath, mode="r")
    
    #artificial indices - serve to recover the whole point information from inFile.points
    artificialIndices = np.arange(len(inFile.x), dtype = int)

    # create an artificial numPy array using x y z classification and index
    coords = np.vstack((inFile.x, inFile.y, inFile.z, inFile.classification, artificialIndices)).transpose()

    # first, sort the 2D NumPy array row-wise so dups will be contiguous
    # and rows are preserved
    a, b, c, d, e = coords.T    # create the keys for to pass to lexsort
    ndx = np.lexsort((a, b, c))

    # replace the array inplace with the ordered sequence
    coords = coords[ndx,]

    # free up some memory (x86 really needs it)
    del ndx
    del a
    del b
    del c
    del d
    del e

    # how many input points
    numRows = coords.shape[0]
    # fake indices pointing towards the initial inFile.points
    indices = np.zeros(numRows, dtype = int)
    duplicates = 0;
    singles = 0;
    idx = 0
    index = 0;
    while (idx < numRows):
        
        jdx = idx + 1;
        singles = singles + 1;
        
        while (jdx < numRows and (coords[idx, 0:3] == coords[jdx, 0:3] ).all() ):
            duplicates = duplicates + 1
            once = True
            if once:
                if (jdx < 1000):
                    print( int(coords[idx][4]), " -- ", coords[idx][0], coords[idx][1], coords[idx][2], coords[idx][3])
                once = False
            jdx = jdx + 1
                    
        indices[index] = int(coords[idx][4])
        index = index + 1
                    
        idx = jdx

    print ("duplicate count = ", duplicates, "single count =", singles)


    del coords # do not need it anymore

    # slice the input points and keep only the ones stored in the indices array
    points_kept = inFile.points[indices[0:index]]



    print("Writing output files...")
    outFile1 = File(out_laspath, mode = "w", header = inFile.header)
    outFile1.points = points_kept
    outFile1.close()
    
    print("Closing input file...")
    inFile.close()