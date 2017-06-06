# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:01:06 2017

@author: ybcheng

from:
http://forum.developer.parrot.com/t/details-of-irradiance-list-tag-for-sunshine-sensor-in-exif-data-of-sequoia/5261/46
"""


import sys
import os
import glob
import exiftool
import base64
import struct













uint64_t timestamp (us)
uint16_t CH0 (count)
uint16_t CH1 (count)
uint16_t gain index
uint16_t integration time (ms)
float    yaw
float    pitch
float    roll


irradiance_list_tag = 'XMP:IrradianceList'
irradiance_calibration_measurement_golden_tag = 'XMP:IrradianceCalibrationMeasurementGolden'
irradiance_calibration_measurement_tag = 'XMP:IrradianceCalibrationMeasurement'

tags = [ irradiance_list_tag, irradiance_calibration_measurement_tag ]

directory = 'test'

channels = [ 'RED', 'NIR' ]

index = 0

for channel in channels:
    files = glob.glob(os.path.join(directory, '*' + channel + '*'))
    with exiftool.ExifTool() as et:
        metadata = et.get_tags_batch(tags, files)
        for file_metadata in metadata:
            irradiance_list = file_metadata[irradiance_list_tag]
            irradiance_calibration_measurement = file_metadata[irradiance_calibration_measurement_tag]

            irradiance_list_bytes = base64.b64decode(irradiance_list)

            print(files[index])
            index += 1

            for irradiance_data in struct.iter_unpack("qHHHHfff", irradiance_list_bytes):
                print(irradiance_data)
