# OwensLakeSPECTIR_v5.py
# J:\Owens_Lake\tasks\ops\SF\SFWCT_Wetness\scripts
# Bronwyn Paxton
# 02 March 2017
# Script to perform wetness analysis on SPECTIR data

# File setup
import arcpy, os
import datetime

t1 = datetime.datetime.now()

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True

from arcpy.sa import *

#######################################################################################################
# Specify basepaths and load input files
#######################################################################################################
# Specify basepaths

# Update these!!
SPECTIR_date = "2017_0224"
SPECTIR_wetness =  r"R:\OwensLake\raster\HyperSpec\20170224\!1600_ASI_OwensLake_XIII\SPECTIR\AS_Delivery\001-015_swir_mosaic_wetness.tif" # SPECTIR imagery
print(SPECTIR_wetness)

basepath_out = r"J:\Owens_Lake\tasks\ops\SF\SFWCT_Wetness" 

basepath_wksp = basepath_out + "\\" + SPECTIR_date
print(basepath_wksp)

basepath_gdb_basefiles = basepath_out + "\\" + r"bndy\SFWCT_Bndys_20170210ver.gdb" # Basefiles
print(basepath_gdb_basefiles)

basepath_gdb_out_name = "SPECTIR_" + SPECTIR_date + ".gdb" # Geodatabase to write files out to
print(basepath_gdb_out_name)

# Load Lateral Areas
LateralAreas = basepath_gdb_basefiles + "\\" + "SFWCT_LateralAreas_T1A4_T10_T26_T29"
LateralAreas_lyr = "LateralAreas" 
arcpy.MakeFeatureLayer_management(LateralAreas, LateralAreas_lyr)

# Load Treatment Areas
TreatmentAreas = basepath_gdb_basefiles + "\\" + "SFWCT_TreatmentAreas_T1A4_T10_T26_T29"
TreatmentAreas_lyr = "TreatmentAreas" 
arcpy.MakeFeatureLayer_management(TreatmentAreas, TreatmentAreas_lyr)

# Load Exclusion Areas
ExclusionAreas = r"J:\Owens_Lake\tasks\ops\SF\0000Template_20131101\For_SF_BaseLayers_For_ABCDGridCreation.gdb\berm_20161012_ExclusionArea_Final_20161012"
ExclusionAreas_lyr = "ExclusionAreas" 
arcpy.MakeFeatureLayer_management(ExclusionAreas, ExclusionAreas_lyr)

# Set up workspace
StartWorkspace = basepath_wksp
arcpy.env.workspace = StartWorkspace

# Set path to (new) file geodatabase
basepath_gdb_out = basepath_wksp + "\\" + basepath_gdb_out_name
print(basepath_gdb_out)


sq = "'"
dq = '"'

RunCreateFishnet = "NO"

if RunCreateFishnet == "YES":
    
    # Create new file geodatabase
    arcpy.CreateFileGDB_management(basepath_wksp, basepath_gdb_out_name)
    
    #######################################################################################################
    # Load wetness raster
    #######################################################################################################
    # Find properties of raster
    SPECTIR_wetness_cellsize_xResult = arcpy.GetRasterProperties_management (SPECTIR_wetness, "CELLSIZEX")
    SPECTIR_wetness_cellsize_x = SPECTIR_wetness_cellsize_xResult.getOutput(0)
    SPECTIR_wetness_cellsize_yResult = arcpy.GetRasterProperties_management (SPECTIR_wetness, "CELLSIZEY")
    SPECTIR_wetness_cellsize_y = SPECTIR_wetness_cellsize_yResult.getOutput(0)
    print(SPECTIR_wetness_cellsize_x)
    print(SPECTIR_wetness_cellsize_y)

    SPECTIR_wetness_cellsize = SPECTIR_wetness_cellsize_x

    #######################################################################################################
    # Loop through DCAs
    #######################################################################################################
    # Specify list of DCAs
    DCAs = ["T29_2", "T10_1", "T26", "T1A_4"]
    cnt0 = 0

    # Step through DCAs
    for DCA in DCAs:
        print(DCA)
        cnt0 = cnt0 + 1

        #######################################################################################################
        # Get pre-defined extent of selected DCA and clip raster
        #######################################################################################################
        if DCA == "T1A_4":
            DCA_extent = str(411700) + " " + str(4023500) + " " + str(414500) + " " + str(4026300)   
        elif DCA == "T10_1":
            DCA_extent = str(416500) + " " + str(4023350) + " " + str(418300) + " " + str(4026050)   
        elif DCA == "T26":
            DCA_extent = str(417900) + " " + str(4036800) + " " + str(421400) + " " + str(4039300)   
        elif DCA == "T29_2":
            DCA_extent = str(416000) + " " + str(4040000) + " " + str(417150) + " " + str(4041400) 
          
        print(DCA_extent)

        SPECTIR_wetness_byDCA = basepath_gdb_out + "\\" + "_0_" + DCA + "_SWIR_mosaic_wetness_clip"
        print(SPECTIR_wetness_byDCA)
        arcpy.Clip_management(in_raster=SPECTIR_wetness, rectangle=DCA_extent, out_raster=SPECTIR_wetness_byDCA, nodata_value="32767", clipping_geometry="NONE", maintain_clipping_extent="NO_MAINTAIN_EXTENT")

        #######################################################################################################
        # Create fishnet from clipped wetness raster
        #######################################################################################################
        SPECTIR_wetness_DCA_xmin = arcpy.GetRasterProperties_management (SPECTIR_wetness_byDCA, "LEFT").getOutput(0)
        SPECTIR_wetness_DCA_xmax = arcpy.GetRasterProperties_management (SPECTIR_wetness_byDCA, "RIGHT").getOutput(0)
        SPECTIR_wetness_DCA_ymin = arcpy.GetRasterProperties_management (SPECTIR_wetness_byDCA, "BOTTOM").getOutput(0)
        SPECTIR_wetness_DCA_ymax = arcpy.GetRasterProperties_management (SPECTIR_wetness_byDCA, "TOP").getOutput(0)
        
        fishnet_DCA = basepath_gdb_out + "\\" + "_1_" + DCA + "_SWIR_mosaic_wetness_net"
        or_coord_DCA = str(SPECTIR_wetness_DCA_xmin) + " " + str(SPECTIR_wetness_DCA_ymin)
        y_coord_DCA = str(SPECTIR_wetness_DCA_xmin) + " " + str(SPECTIR_wetness_DCA_ymax)
        cor_coord_DCA = str(SPECTIR_wetness_DCA_xmax) + " " + str(SPECTIR_wetness_DCA_ymax)
        arcpy.CreateFishnet_management(out_feature_class=fishnet_DCA, origin_coord=or_coord_DCA, y_axis_coord=y_coord_DCA, cell_width=SPECTIR_wetness_cellsize, cell_height=SPECTIR_wetness_cellsize, number_rows="", number_columns="", corner_coord=cor_coord_DCA, labels="LABELS", template=SPECTIR_wetness_byDCA, geometry_type="POLYGON")    

        #######################################################################################################
        # Extract wetness values to points
        #######################################################################################################
        fishnet_DCA_labels = basepath_gdb_out + "\\" + "_1_" + DCA + "_SWIR_mosaic_wetness_net_label"
        fishnet_DCA_labels_wetval = basepath_gdb_out + "\\" + "_1_" + DCA + "_SWIR_mosaic_wetness_net_label_wetval"
        arcpy.sa.ExtractValuesToPoints(fishnet_DCA_labels, SPECTIR_wetness_byDCA, fishnet_DCA_labels_wetval, "NONE", "VALUE_ONLY")

        #######################################################################################################
        # Join points to polygons
        #######################################################################################################
        fishnet_DCA_wetval = basepath_gdb_out + "\\" + "_2_" + DCA + "_SWIR_mosaic_wetness_net_wetval"
        arcpy.SpatialJoin_analysis(target_features=fishnet_DCA, join_features=fishnet_DCA_labels_wetval, out_feature_class=fishnet_DCA_wetval, join_operation="JOIN_ONE_TO_ONE", join_type="KEEP_ALL", match_option="INTERSECT", search_radius="", distance_field_name="")

        #######################################################################################################
        # Delete extra fields
        #######################################################################################################
        drop_fields = ["Join_Count", "TARGET_FID"]
        arcpy.DeleteField_management(fishnet_DCA_wetval, drop_fields)

        #######################################################################################################
        # Append fishnets
        #######################################################################################################
        if cnt0 == 1:
            master_fishnet_DCA = basepath_gdb_out + "\\" + "_3_master_SWIR_mosaic_wetness_net"
            arcpy.CopyFeatures_management(fishnet_DCA_wetval, master_fishnet_DCA)
        else:
            arcpy.Append_management(fishnet_DCA_wetval, master_fishnet_DCA)

else:
    # Load master fishnet
    master_fishnet_DCA = basepath_gdb_out + "\\" + "_3_master_SWIR_mosaic_wetness_net"

#######################################################################################################
# Set up flag fields for treatment areas and lateral areas
#######################################################################################################
print("Add Flags to Fishnet")

master_fishnet_DCA_lyr = "fishnet_DCA_lyr"
arcpy.MakeFeatureLayer_management(master_fishnet_DCA, master_fishnet_DCA_lyr)

if RunCreateFishnet == "YES":
    arcpy.AddField_management(in_table=master_fishnet_DCA_lyr, field_name="Treatment", field_type="TEXT", field_length=20)
    arcpy.AddField_management(in_table=master_fishnet_DCA_lyr, field_name="Lateral", field_type="TEXT", field_length=20)
    
arcpy.CalculateField_management (in_table=master_fishnet_DCA_lyr, field="Treatment", expression='"N"', expression_type="PYTHON")
arcpy.CalculateField_management (in_table=master_fishnet_DCA_lyr, field="Lateral", expression='"N"', expression_type="PYTHON")

#######################################################################################################
# Select features within Lateral Areas
#######################################################################################################
print("Select features within Lateral Areas")
cursor_lats = arcpy.da.SearchCursor(LateralAreas_lyr, ['DCA_Lateral'])

for row_lats in cursor_lats:
    print(row_lats[0])
    arcpy.SelectLayerByAttribute_management(LateralAreas_lyr, "NEW_SELECTION", "DCA_Lateral = " + sq + str(row_lats[0]) + sq)
    arcpy.SelectLayerByLocation_management(master_fishnet_DCA_lyr, "HAVE_THEIR_CENTER_IN", LateralAreas_lyr, "", "NEW_SELECTION")
    arcpy.SelectLayerByLocation_management(master_fishnet_DCA_lyr, "INTERSECT", ExclusionAreas_lyr, "", "REMOVE_FROM_SELECTION")
    arcpy.CalculateField_management(master_fishnet_DCA_lyr, "Lateral", dq + str(row_lats[0]) + dq, "PYTHON", "")

arcpy.SelectLayerByAttribute_management(master_fishnet_DCA_lyr, "CLEAR_SELECTION")

#######################################################################################################
# Select features within Treatment Areas
#######################################################################################################
print("Select features within Treatment Areas")
cursor_tas = arcpy.da.SearchCursor(TreatmentAreas_lyr, ['HYDROGRP'])

for row_tas in cursor_tas:
    print(row_tas[0])
    arcpy.SelectLayerByAttribute_management(TreatmentAreas_lyr, "NEW_SELECTION", "HYDROGRP = " + sq + str(row_tas[0]) + sq)
    arcpy.SelectLayerByLocation_management(master_fishnet_DCA_lyr, "HAVE_THEIR_CENTER_IN", TreatmentAreas_lyr, "", "NEW_SELECTION")
    arcpy.SelectLayerByLocation_management(master_fishnet_DCA_lyr, "INTERSECT", ExclusionAreas_lyr, "", "REMOVE_FROM_SELECTION")
    arcpy.CalculateField_management(master_fishnet_DCA_lyr, "Treatment", dq + str(row_tas[0]) + dq, "PYTHON", "")

arcpy.SelectLayerByAttribute_management(master_fishnet_DCA_lyr, "CLEAR_SELECTION")

#######################################################################################################
## Subset polygons and run statistics by treatment areas and laterals
#######################################################################################################
print("Run statistics")
stats_lyr = "stats_lyr"
arcpy.MakeFeatureLayer_management(master_fishnet_DCA, stats_lyr, "Treatment <> 'N' OR Lateral <> 'N'")

# Total count by treatment areas
Stats_by_ta_total_cnt = basepath_gdb_out + "\\" + "_4_pnts_stats_totalcnt_by_treatmentarea"
arcpy.Statistics_analysis(in_table=stats_lyr, out_table=Stats_by_ta_total_cnt, statistics_fields="RASTERVALU COUNT", case_field="Treatment")
Stats_by_ta_total_cnt_tbvw = "_6_stats_by_treatmentarea"
arcpy.MakeTableView_management(Stats_by_ta_total_cnt, Stats_by_ta_total_cnt_tbvw)

# Total count by lateral area
Stats_by_lat_total_cnt = basepath_gdb_out + "\\" + "_4_pnts_stats_totalcnt_by_lateral"
arcpy.Statistics_analysis(in_table=stats_lyr, out_table=Stats_by_lat_total_cnt, statistics_fields="RASTERVALU COUNT", case_field="Lateral")
Stats_by_lat_total_cnt_tbvw = "_6_stats_by_lateral"
arcpy.MakeTableView_management(Stats_by_lat_total_cnt, Stats_by_lat_total_cnt_tbvw)

# Wetness by treatment area
arcpy.SelectLayerByAttribute_management(in_layer_or_view=stats_lyr, selection_type="NEW_SELECTION", where_clause="RASTERVALU = 100")
Stats_by_ta_wetness = basepath_gdb_out + "\\" + "_5_pnts_stats_wetness_by_treatmentarea"
arcpy.Statistics_analysis(in_table=stats_lyr, out_table=Stats_by_ta_wetness, statistics_fields="RASTERVALU COUNT", case_field="Treatment;RASTERVALU")
Stats_by_ta_wetness_tbvw = "_5_pnts_stats_wetness_by_treatmentarea"
arcpy.MakeTableView_management(Stats_by_ta_wetness, Stats_by_ta_wetness_tbvw)

# Wetness by lateral area
arcpy.SelectLayerByAttribute_management(in_layer_or_view=stats_lyr, selection_type="NEW_SELECTION", where_clause="RASTERVALU = 100")
Stats_by_lat_wetness = basepath_gdb_out + "\\" + "_5_pnts_stats_wetness_by_lateral"
arcpy.Statistics_analysis(in_table=stats_lyr, out_table=Stats_by_lat_wetness, statistics_fields="RASTERVALU COUNT", case_field="Lateral;RASTERVALU")
Stats_by_lat_wetness_tbvw = "_5_pnts_stats_wetness_by_treatmentarea"
arcpy.MakeTableView_management(Stats_by_lat_wetness, Stats_by_lat_wetness_tbvw)

# Join treatment area tables
arcpy.AddJoin_management (in_layer_or_view=Stats_by_ta_total_cnt_tbvw, in_field="Treatment", join_table=Stats_by_ta_wetness_tbvw, join_field="Treatment")
arcpy.TableToGeodatabase_conversion (Stats_by_ta_total_cnt_tbvw, basepath_gdb_out)

# Join lateral area tables
arcpy.AddJoin_management (in_layer_or_view=Stats_by_lat_total_cnt_tbvw, in_field="Lateral", join_table=Stats_by_lat_wetness_tbvw, join_field="Lateral")
arcpy.TableToGeodatabase_conversion (Stats_by_lat_total_cnt_tbvw, basepath_gdb_out)

#######################################################################################################
# Convert tables to Excel
#######################################################################################################
print("Create Excel tables")
# By treatment areas
xls_stats_by_ta = basepath_wksp + "\\" + "SPECTIR_stats_by_treatmentarea.xls"
arcpy.TableToExcel_conversion(Input_Table=Stats_by_ta_total_cnt_tbvw, Output_Excel_File=xls_stats_by_ta, Use_field_alias_as_column_header="NAME", Use_domain_and_subtype_description="CODE")

# By laterals
xls_stats_by_lat = basepath_wksp + "\\" + "SPECTIR_stats_by_lateral.xls"
arcpy.TableToExcel_conversion(Input_Table=Stats_by_lat_total_cnt_tbvw, Output_Excel_File=xls_stats_by_lat, Use_field_alias_as_column_header="NAME", Use_domain_and_subtype_description="CODE")

t2 = datetime.datetime.now()
print(str(t2-t1))
print "Done!"
print
