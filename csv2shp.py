# -*- coding: utf-8 -*-

"""
grassland mowing events detection

Author:
Date:
"""
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def csv2shp(in_path, out_path, field_xx='POINT_X', field_yy='POINT_Y'):

    csv_df = pd.read_csv(in_path, header=0, encoding='gbk')
    geometry = [Point(xy) for xy in zip(csv_df[field_xx], csv_df[field_yy])]
    gdf = gpd.GeoDataFrame(csv_df, crs="EPSG:4326", geometry=geometry)
    gdf.to_file(out_path, encoding='gbk')

    return out_path


def main():
    print("###########################################################")
    print("### ***            ########################################")
    print("###########################################################")

    csv_path = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X22Y09\csv\MODIS_mowing.csv'
    shp_path = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X22Y09\csv\MODIS_mowing.shp'

    csv2shp(csv_path, shp_path)

    print("### Task over #############################################")


if __name__ == "__main__":
    main()
