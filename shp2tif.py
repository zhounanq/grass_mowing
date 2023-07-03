# -*- coding: utf-8 -*-
"""

Author:
Date:
"""
import os
import argparse
import arcpy


def point2raster(in_features, out_rasterdataset, value_field=None, cell_assignment=None, priority_field=None, reference_raster=None):

    # create dst folder is not exists
    dir_name = os.path.dirname(out_rasterdataset)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Set environment setting
    arcpy.env.extent = reference_raster
    # arcpy.env.snapRaster = reference_raster

    # description = arcpy.Describe(reference_raster)
    # cellsize = description.children[0].meanCellHeight

    # Set local variables
    inFeatures = in_features
    valField = value_field
    outRaster = out_rasterdataset
    assignmentType = "MOST_FREQUENT"
    priorityField = "NONE"
    cellSize = reference_raster

    # Execute PointToRaster
    arcpy.PointToRaster_conversion(inFeatures, valField, outRaster, assignmentType, priorityField, cellSize)
    print arcpy.GetMessages(0)

    return out_rasterdataset


def main():
    print("###########################################################")
    print("### ***            ########################################")
    print("###########################################################")

    in_features = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X18Y05\mowing\MODIS_mowing.shp'
    out_raster = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X18Y05\mowing\MODIS_mowing_sum.tif'
    ref_raster = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X18Y05\ref\PROBAV_S10_TOC_X18Y05_1KM_V101.HDF5.tif'
    point2raster(in_features, out_raster, value_field='sum', reference_raster=ref_raster)

    print("### Task over #############################################")


if __name__ == "__main__":
    main()
