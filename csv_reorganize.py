# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import datetime
import numpy as np
import pandas as pd


def csv_cross_merge(srcfile, targetfile, header_cols=5, header_rows=None):

    src_array = pd.read_csv(srcfile, header=header_rows)

    src_header_cols = src_array.iloc[:, :header_cols]
    src_data_cols = src_array.iloc[:, header_cols:]

    data_shape = src_data_cols.shape
    half_num_col = int(data_shape[1]/2)

    target_array = pd.DataFrame(data=src_header_cols)

    for cc in range(0, half_num_col):
        target_array = pd.concat([target_array, src_data_cols.iloc[:, cc], src_data_cols.iloc[:, cc+half_num_col]], axis=1)

    target_array.to_csv(targetfile, header=False, index=False)


def main():
    now = datetime.datetime.now()
    print("###########################################################")
    print("### ***            ########################################")
    print("### ", now)
    print("###########################################################")

    srcfile = r'I:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X21Y07\csv\csv_vits_MODIS_ndvi.csv'
    targetfile = r'I:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X21Y07\csv\csv_vits_MODIS_ndvi_c.csv'
    csv_cross_merge(srcfile, targetfile)


    print("### Task over #############################################")


if __name__ == "__main__":
    main()
