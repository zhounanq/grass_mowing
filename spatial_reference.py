# -*- coding: utf-8 -*-

"""
Image operations

Author: Zhou Ya'nan
"""
import os
import argparse


from osgeo import gdal, osr


def parse_args():
    parser = argparse.ArgumentParser(description='Copy spatial reference for target raster dataset')
    parser.add_argument('--georef-file', required=False, type=str, default="./georef.tif",
                        help='spatial reference for results')
    parser.add_argument('--target-file', required=False, type=str, default="./target.tif",
                        help='target file')
    opts = parser.parse_args()
    return opts


def copy_spatialref(img_path4, img_path2):
    image_ds4 = gdal.Open(img_path4, gdal.GA_ReadOnly)
    if not image_ds4:
        print("Fail to open image {}".format(img_path4))
        return False

    image_ds2 = gdal.Open(img_path2, gdal.GA_Update)
    if not image_ds2:
        print("Fail to open image {}".format(img_path2))
        return False

    proj = image_ds4.GetProjection()
    if proj:
        print("Projection is {}".format(proj))
    geotransform = image_ds4.GetGeoTransform()
    if geotransform:
        print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

    image_ds2.SetProjection(proj)
    image_ds2.SetGeoTransform(geotransform)

    print("### Copy spatial reference over")
    return True


def main():
    print("###########################################################")
    print("### Spatial Reference #####################################")
    print("###########################################################")

    # parameters
    opts = parse_args()
    georef_file = opts.georef_file
    target_file = opts.target_file

    img_path4 = georef_file
    img_path2 = target_file
    # img_path4 = r"I:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X17Y06\spatialref\PROBAV_S10_TOC_X17Y06_20200101_1KM_NDVI_V101_NDVI.tif"
    # img_path2 = r"I:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X17Y06\2020_EVI\PROBAV_S10_TOC_X17Y06_20200101_1KM_V101_EVI.tif"

    # run script
    copy_spatialref(img_path4, img_path2)

    print("### Task over #############################################")


if __name__ == "__main__":
    main()
