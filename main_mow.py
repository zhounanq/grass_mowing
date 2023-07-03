# -*- coding: utf-8 -*-

"""
grassland mowing events detection

Author:
Date:
"""
import os

from mowing_detection import *
from postprocessing import ensemble_detected_events_voting
from csv2shp import csv2shp


def main():
    print("###########################################################")
    print("### ***            ########################################")
    print("###########################################################")

    ndvi_path = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X18Y05\csv\MODIS_ndvi.csv'
    ndvi_result = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X18Y05\csv\MODIS_ndvi_mowing.csv'
    main_XXXYYYY(ndvi_path, ndvi_result, 19)

    evi_path = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X18Y05\csv\MODIS_evi.csv'
    evi_result = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X18Y05\csv\MODIS_evi_mowing.csv'
    main_XXXYYYY(evi_path, evi_result, 19)

    ensemble_result = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X18Y05\csv\MODIS_mowing.csv'
    ensemble_detected_events_voting(ndvi_result, evi_result, ensemble_result)

    shp_path = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X18Y05\csv\MODIS_mowing.shp'
    csv2shp(ensemble_result, shp_path)

    print("### Task over #############################################")




if __name__ == "__main__":
    main()
