"""
"""

import sys
import os
import time

import rasterio
import cv2
import numpy as np
from numpy.linalg.linalg import LinAlgError

from ..imops import imio
from ..dbops import finder, parse
from ..gen import fileio, wrappers, dirfuncs, strops, camsettings, osops
from ..cv import refutils
from ..gis import rastertools, shapetools

CAM_PARAMS = camsettings.PARAMS
DROPBOX_DIR = dirfuncs.guess_dropbox_dir()
if DROPBOX_DIR is not None:
    EXECUTABLE_DIR = DROPBOX_DIR + 'Executables/'


def align_ids_fields(filepaths, dummy=None, replace=False):
    """
    Wrapper for executable wrapper, to take all files in a directory.
    Assumes that pairs of Red/NIR images exist, and then runs on each pair.
    If the field IDs do not match, will not run, but possible errors still
    if subfields are present.

    Parameters
    ----------
    filepaths : list
        List of strings with full paths to NIR/IDS images.

    Returns
    -------
    None
    """

    tifs = [f for f in filepaths if f.endswith('.tif') and 'auto' not in f]
    nir_files = [f for f in tifs if 'ids nir' in f.lower()]
    #refs = [r for r in refpaths if r.endswith('.tif')]
    #ref_files = [r for r in refs if 'ids5' in r.lower()]
    
    # TODO: smarter matching without looking at everything
    #if red_files or nir_files:
    #    longer = max(red_files, nir_files, key=len)
    #    search_dir = os.path.dirname(longer[0]) + '/'
    #    red_files = fileio.insensitive_glob(search_dir + '*ids red.tif')
    #    nir_files = fileio.insensitive_glob(search_dir + '*ids nir.tif')

    # look through all files with lots of checks to make sure we are not
    # overwriting anything
    #for red in red_files: 
    #    red_fid = parse.get_fid_from_filename(red)
    for nir in nir_files:
        ids_filename = strops.ireplace('IDS NIR', 'IDS5', nir)
        if not os.path.exists(ids_filename) or replace:
            if (not nir == ids_filename):
                print("Generating %s" % ids_filename)
                align_ids_field(nir, ids_filename)


def align_ids_field(filepath1, outpath, pre_mask=True, num_chan=5):
    """
    Wrapper for executable alignment script.

    Parameters
    ----------
    filepath1 : str
        Full path to nir channel image
    outpath : str
        Filename of output stacked IDS image.
    
    Returns
    -------
    None
    """

    start_dir = os.path.abspath(os.path.curdir)
    filepath1 = filepath1.replace('\\', '/')
    #filepath2 = filepath2.replace('\\', '/')
    filepath2 = filepath1.replace('NIR', 'Red')
    filepath3 = filepath1.replace('NIR', '700')
    filepath4 = filepath1.replace('NIR', '550')
    filepath5 = filepath1.replace('NIR', '480')
    outpath = outpath.replace('\\', '/')

    if pre_mask:
        masked1 = pre_mask_image(filepath1)
        masked2 = pre_mask_image(filepath2)
        masked3 = pre_mask_image(filepath3)
        masked4 = pre_mask_image(filepath4)
        masked5 = pre_mask_image(filepath5)
        if ((masked1 != '') and (masked2 != '') and (masked3 != '') and
            (masked4 != '') and (masked5 != '')):
            filepath1 = start_dir + '/' + masked1
            filepath2 = start_dir + '/' + masked2
            filepath3 = start_dir + '/' + masked3
            filepath4 = start_dir + '/' + masked4
            filepath5 = start_dir + '/' + masked5

    exe_path = EXECUTABLE_DIR + "alignMosaicIDS.exe"
    call = [exe_path, filepath1, "", outpath, "2", "5"]

    os.chdir(EXECUTABLE_DIR)
    print(" ".join(call))
    sys.stdout.flush()
    error_str = "Error aligning ids files %s" % (filepath1)
    osops.syscall(call, error_str=error_str)
    os.chdir(start_dir)
    sys.stdout.flush()

    # clean up
    if os.path.dirname(filepath1) == start_dir and pre_mask:
        os.remove(filepath1)
        os.remove(filepath2)
        os.remove(filepath3)
        os.remove(filepath4)
        os.remove(filepath5)


# function to automatically apply referencing function as files appear
align_watch = wrappers.gen_watcher(align_ids_fields, wrappers.gen_applier,
                                   wkwargs=dict(change_delay=30))


def pre_mask_image(input_filepath, buf_size=0.0015):
                        #same_buf=False):
    """
    To improve upon the likelihood of success in register, pre-mask the
    reference image and mosaic to a small buffer around the area of interest.
    Write new files to be used for registration.

    Parameters
    ----------
    input_filepath : str
        Full path to file to be registered
    reference_filepath : str
        Full path to reference file
    buf_size : float (opt)
        Defaults to 0.0015. This is in units of degrees, since that is the
        coordinate system in use here, but would be good to find a more
        general workaround.
    same_buf : bool (opt)
        Whether to apply the same masking buffer to the reference image.
        Default is False, in which case half the buffer size is used.

    Parameters
    ----------
    local_input : str
        Path to a locally saved version of input file, with a generous
        mask applied.
    local_reference : str
        Path to a locally saved version of the reference file, with a generous
        mask applied.
    """

    fid = parse.get_fid_from_filename(input_filepath)
    sfid = parse.guess_sfid_from_filename(input_filepath)
    shapefilename, exists = finder.get_field_shapefile(fid)

    # TODO: replace shapefile generation with masking by polygon
    if sfid is not None:
        split_shapes = shapetools.gen_block_shapefiles(shapefilename,
                                                       key_name="mosaic")
        field_name = os.path.basename(shapefilename).split('.')[0] + ' ' + sfid
        for ss in split_shapes:
            if field_name in ss:
                shapefilename = ss
            else:
                fileio.cleanup_file_group(os.path.curdir + '/',
                                          ss.split('.')[0])

    if not exists:
        return "", ""

    #if same_buf:
    #    buf_size_ref = buf_size
    #else:
    #    buf_size_ref = buf_size / 2.

    local_input = os.path.basename(input_filepath)
    #local_reference = os.path.basename(reference_filepath)
    #local_reference = local_reference.replace('.tif', ' ref.tif')
    # better to handle a partial file and not pass it on to the next
    # step, rather than register something we know will fail, but using
    # skip_partial=False is better to at least keep the loop going
    rastertools.retrier(rastertools.mask_by_shapefile, input_filepath,
                        shapefilename, local_input, buf_size=buf_size,
                        skip_partial=False, mask_val=0)
    #rastertools.retrier(rastertools.mask_by_shapefile, reference_filepath,
    #                    shapefilename, local_reference, buf_size=buf_size_ref,
    #                    skip_partial=False, mask_val=0)

    return local_input

