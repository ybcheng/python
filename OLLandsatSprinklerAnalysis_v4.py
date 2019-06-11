# OwensLakeLandsatSprinklerAnalysis_v4.py
# J:\Owens_Lake\tasks\ops\SF\SFWCT_Wetness\scripts
# Bronwyn Paxton
# 10 February 2017
# Script to perform wetness analysis
# Assumes no lateral areas do not overlap other lateral areas (likewise for treatment areas)

# File setup
import arcpy, os

#######################################################################################################
# Specify basepaths and load input files
#######################################################################################################

# Specify basepaths
basepath_out = r"J:\Owens_Lake\tasks\ops\SF"
Landsat_date = "2017_0305"
Landsat_type = "L8"
basepath_gdb_out_name = Landsat_type + "_" + Landsat_date + ".gdb" # Geodatabase to write files out to
print(basepath_gdb_out_name)
basepath_gdb_basefiles = basepath_out + "\\" + r"SFWCT_Wetness\bndy\SFWCT_Bndys_20170210ver.gdb" # Basefiles
print(basepath_gdb_basefiles)
basepath_gdb_Landsat = basepath_out + "\\" + Landsat_date + "\\" + "SFWetnessReporting.gdb" # Landsat analysis
print(basepath_gdb_Landsat)
basepath_wksp = basepath_out + "\\" + "SFWCT_Wetness" + "\\" + Landsat_date + "_" + Landsat_type
print(basepath_wksp)

# Set up workspace
StartWorkspace = basepath_wksp
   
arcpy.env.workspace = StartWorkspace
arcpy.env.overwriteOutput = True

# Create new file geodatabase
basepath_gdb_out = basepath_wksp + "\\" + basepath_gdb_out_name
print(basepath_gdb_out)
arcpy.CreateFileGDB_management(basepath_wksp, basepath_gdb_out_name)

#######################################################################################################
# Load wetness and standing water data
#######################################################################################################

# Load wetness data
WetnessData_a = basepath_gdb_Landsat + r"\Q_SFextract_Wet_a_" + Landsat_date
WetnessData_a_lyr = "WetnessData_a"
arcpy.MakeFeatureLayer_management(WetnessData_a, WetnessData_a_lyr)

#WetnessData_b = basepath_gdb_Landsat + r"\Q_SFextract_Wet_b_" + Landsat_date
#WetnessData_b_lyr = "WetnessData_b"
#arcpy.MakeFeatureLayer_management(WetnessData_b, WetnessData_b_lyr)

# Load standing water data
StandingWaterData = basepath_gdb_Landsat + r"\F_SFextract_SW_" + Landsat_date
StandingWaterData_lyr = "StandingWaterData"
arcpy.MakeFeatureLayer_management(StandingWaterData, StandingWaterData_lyr)

#######################################################################################################
# Spatial join wetness and standing water data
#######################################################################################################
Wet_SW_Data = basepath_gdb_out + "\\" +  "_0_" + Landsat_date + "_QWet_FSW_SFextract"
arcpy.SpatialJoin_analysis(target_features=WetnessData_a_lyr, join_features=StandingWaterData_lyr, out_feature_class=Wet_SW_Data, join_operation="JOIN_ONE_TO_ONE", join_type="KEEP_ALL", match_option="HAVE_THEIR_CENTER_IN", search_radius="", distance_field_name="")

#######################################################################################################
# Delete extra fields
#######################################################################################################
drop_fields_1 = ["Join_Count", "TARGET_FID", "Join_Count_1", "TARGET_FID_1", "Join_Count_12", "TARGET_FID_12",
                 "FEID_1", "HYDROGRP_1", "Count_1", "Count", "ID", "ID_1", "BLOCKID", "BLOCKID_1", "Area_m2", "Area_m2_1",
                 "SLC", "VALUE2", "SLC_1", "VALUE2_1", "ImgAcqu_Date_1", "FEID", "ImgAcqu_Date"]
arcpy.DeleteField_management(Wet_SW_Data, drop_fields_1)

#######################################################################################################
# Load laterals and treatment areas
#######################################################################################################
# Load Lateral Areas
LateralAreas = basepath_gdb_basefiles + "\\" + "SFWCT_LateralAreas_T1A4_T10_T26_T29"
LateralAreas_lyr = "LateralAreas" 
arcpy.MakeFeatureLayer_management(LateralAreas, LateralAreas_lyr)

# Load Treatment Areas
TreatmentAreas = basepath_gdb_basefiles + "\\" + "SFWCT_TreatmentAreas_T1A4_T10_T26_T29"
TreatmentAreas_lyr = "TreatmentAreas" 
arcpy.MakeFeatureLayer_management(TreatmentAreas, TreatmentAreas_lyr)

#######################################################################################################
# Spatial join to add treatment and lateral areas and delete extra fields
#######################################################################################################
DCA_lats = basepath_gdb_out + "\\" + "_1_" + Landsat_date + "_QWet_FSW_SFextract_lats"
arcpy.SpatialJoin_analysis(target_features=Wet_SW_Data, join_features=LateralAreas_lyr, out_feature_class=DCA_lats, join_operation="JOIN_ONE_TO_ONE", join_type="KEEP_ALL", match_option="HAVE_THEIR_CENTER_IN", search_radius="", distance_field_name="")

drop_fields_2 = ["Join_Count", "TARGET_FID", "HYDROGRP_1", "DCA", "TargetWet"]
arcpy.DeleteField_management(DCA_lats, drop_fields_2)

DCA_lats_ta = basepath_gdb_out + "\\" + "_2_" + Landsat_date + "_QWet_FSW_SFextract_lats_ta"
arcpy.SpatialJoin_analysis(target_features=DCA_lats, join_features=TreatmentAreas_lyr, out_feature_class=DCA_lats_ta, join_operation="JOIN_ONE_TO_ONE", join_type="KEEP_ALL", match_option="HAVE_THEIR_CENTER_IN", search_radius="", distance_field_name="")

drop_fields_3 = ["Join_Count", "HYDROGRP", "Name_BACM", "Code", "HYDROGRPorig", "TARGET_FID", "DCA", "BACM_Type"]
arcpy.DeleteField_management(DCA_lats_ta, drop_fields_3)

#########################################################################################################
### Run statistics by treatment areas and laterals
#########################################################################################################
DCA_lats_ta_lyr = "DCA_lats_ta_lyr"
arcpy.MakeFeatureLayer_management(DCA_lats_ta, DCA_lats_ta_lyr)

# Total count by treatment areas
Stats_by_ta_total_cnt = basepath_gdb_out + "\\" + "_3_QWet_FSW_SFextract_lats_ta_stats_by_treatmentarea_totalcnt"
arcpy.Statistics_analysis(in_table=DCA_lats_ta_lyr, out_table=Stats_by_ta_total_cnt, statistics_fields="RASTERVALU COUNT", case_field="HYDROGRP_1")
Stats_by_ta_total_cnt_tbvw = "_6_QWet_FSW_SFextract_lats_ta_stats_by_treatmentarea"
arcpy.MakeTableView_management(Stats_by_ta_total_cnt, Stats_by_ta_total_cnt_tbvw)

# Total count by lateral area
Stats_by_lat_total_cnt = basepath_gdb_out + "\\" + "_3_QWet_FSW_SFextract_lats_ta_stats_by_lateral_totalcnt"
arcpy.Statistics_analysis(in_table=DCA_lats_ta_lyr, out_table=Stats_by_lat_total_cnt, statistics_fields="RASTERVALU COUNT", case_field="DCA_Lateral")
Stats_by_lat_total_cnt_tbvw = "_6_QWet_FSW_SFextract_lats_ta_stats_by_lateral"
arcpy.MakeTableView_management(Stats_by_lat_total_cnt, Stats_by_lat_total_cnt_tbvw)

# Wetness by treatment area
arcpy.SelectLayerByAttribute_management(in_layer_or_view=DCA_lats_ta_lyr, selection_type="NEW_SELECTION", where_clause="WETBINARY = 100")
Stats_by_ta_wetness = basepath_gdb_out + "\\" + "_4_QWet_FSW_SFextract_lats_ta_stats_by_treatmentarea_wetness"
arcpy.Statistics_analysis(in_table=DCA_lats_ta_lyr, out_table=Stats_by_ta_wetness, statistics_fields="WETBINARY COUNT", case_field="HYDROGRP_1;WETBINARY")
Stats_by_ta_wetness_tbvw = "_04_QWet_FSW_SFextract_lats_ta_stats_by_treatmentarea_wetness"
arcpy.MakeTableView_management(Stats_by_ta_wetness, Stats_by_ta_wetness_tbvw)

# Wetness by lateral area
arcpy.SelectLayerByAttribute_management(in_layer_or_view=DCA_lats_ta_lyr, selection_type="NEW_SELECTION", where_clause="WETBINARY = 100")
Stats_by_lat_wetness = basepath_gdb_out + "\\" + "_4_QWet_FSW_SFextract_lats_ta_stats_by_lateral_wetness"
arcpy.Statistics_analysis(in_table=DCA_lats_ta_lyr, out_table=Stats_by_lat_wetness, statistics_fields="WETBINARY COUNT", case_field="DCA_Lateral;WETBINARY")
Stats_by_lat_wetness_tbvw = "_04_QWet_FSW_SFextract_lats_ta_stats_by_lateral_wetness"
arcpy.MakeTableView_management(Stats_by_lat_wetness, Stats_by_lat_wetness_tbvw)

# Standing water by treatment area
arcpy.SelectLayerByAttribute_management(in_layer_or_view=DCA_lats_ta_lyr, selection_type="NEW_SELECTION", where_clause="RASTERVALU = 100")
Stats_by_ta_standingwater = basepath_gdb_out + "\\" + "_5_QWet_FSW_SFextract_lats_ta_stats_by_treatmentarea_standingwater"
arcpy.Statistics_analysis(in_table=DCA_lats_ta_lyr, out_table=Stats_by_ta_standingwater, statistics_fields="RASTERVALU COUNT", case_field="HYDROGRP_1;RASTERVALU")
Stats_by_ta_standingwater_tbvw = "_05_QWet_FSW_SFextract_lats_ta_stats_by_treatmentarea_standingwater"
arcpy.MakeTableView_management(Stats_by_ta_standingwater, Stats_by_ta_standingwater_tbvw)

# Standing water by lateral
arcpy.SelectLayerByAttribute_management(in_layer_or_view=DCA_lats_ta_lyr, selection_type="NEW_SELECTION", where_clause="RASTERVALU = 100")
Stats_by_lat_standingwater = basepath_gdb_out + "\\" + "_5_QWet_FSW_SFextract_lats_ta_stats_by_lateral_standingwater"
arcpy.Statistics_analysis(in_table=DCA_lats_ta_lyr, out_table=Stats_by_lat_standingwater, statistics_fields="RASTERVALU COUNT", case_field="DCA_Lateral;RASTERVALU")
Stats_by_lat_standingwater_tbvw = "_05_QWet_FSW_SFextract_lats_ta_stats_by_lateral_standingwater"
arcpy.MakeTableView_management(Stats_by_lat_standingwater, Stats_by_lat_standingwater_tbvw)

# Join treatment area tables
arcpy.AddJoin_management (in_layer_or_view=Stats_by_ta_total_cnt_tbvw, in_field="HYDROGRP_1", join_table=Stats_by_ta_wetness_tbvw, join_field="HYDROGRP_1")
arcpy.AddJoin_management (in_layer_or_view=Stats_by_ta_total_cnt_tbvw, in_field="HYDROGRP_1", join_table=Stats_by_ta_standingwater_tbvw, join_field="HYDROGRP_1")
arcpy.TableToGeodatabase_conversion (Stats_by_ta_total_cnt_tbvw, basepath_gdb_out)

# Join lateral area tables
arcpy.AddJoin_management (in_layer_or_view=Stats_by_lat_total_cnt_tbvw, in_field="DCA_Lateral", join_table=Stats_by_lat_wetness_tbvw, join_field="DCA_Lateral")
arcpy.AddJoin_management (in_layer_or_view=Stats_by_lat_total_cnt_tbvw, in_field="DCA_Lateral", join_table=Stats_by_lat_standingwater_tbvw, join_field="DCA_Lateral")
arcpy.TableToGeodatabase_conversion (Stats_by_lat_total_cnt_tbvw, basepath_gdb_out)

#######################################################################################################
# Convert tables to Excel
#######################################################################################################
# By treatment areas
xls_stats_by_ta = basepath_wksp + "\\" + "Landsat_stats_by_treatmentarea.xls"
arcpy.TableToExcel_conversion(Input_Table=Stats_by_ta_total_cnt_tbvw, Output_Excel_File=xls_stats_by_ta, Use_field_alias_as_column_header="NAME", Use_domain_and_subtype_description="CODE")

# By laterals
xls_stats_by_lat = basepath_wksp + "\\" + "Landsat_stats_by_lateral.xls"
arcpy.TableToExcel_conversion(Input_Table=Stats_by_lat_total_cnt_tbvw, Output_Excel_File=xls_stats_by_lat, Use_field_alias_as_column_header="NAME", Use_domain_and_subtype_description="CODE")

print "Done!"
print
