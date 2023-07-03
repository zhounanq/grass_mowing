# -*- coding: utf-8 -*-

"""
grassland mowing events detection

Author:
Date:
"""

import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, signal


def ensemble_detected_events_voting(event_file1, event_file2, ensemble_file):

    # opening two files
    event_data1 = pd.read_csv(event_file1, header=0)
    event_header1 = event_data1.iloc[:, :5]
    event_array1 = event_data1.iloc[:, 5:]

    event_data2 = pd.read_csv(event_file2, header=0)
    event_header2 = event_data1.iloc[:, :5]
    event_array2 = event_data1.iloc[:, 5:]
    assert(event_array1.shape[0] == event_array1.shape[0])

    (row, col) = event_array1.shape
    print("### Total row {}, and total time step {}".format(row, col))

    # ensembling
    ensemble_event_array = np.zeros([row, col], dtype='int')
    ensemble_event_num = np.zeros([row, 1], dtype='int')
    for rr in range(0, row):

        events1, events2 = np.array(event_array1.iloc[rr]), np.array(event_array2.iloc[rr])

        event_intersect = np.intersect1d(events1, events2)
        event_intersect = np.sort(event_intersect)
        event_intersect = np.pad(event_intersect, (0, col - len(event_intersect)), 'constant', constant_values=(0, 0))
        event_num = sum(event_intersect != 0)

        ensemble_event_array[rr] = event_intersect
        ensemble_event_num[rr][0] = event_num
        print("ROW {}: Mowing events {} @ {}".format(rr, event_num, event_intersect))
    # for

    # writing
    event_df = pd.DataFrame(ensemble_event_array, columns=["mow_{}".format(ii+1) for ii in range(0, col)])
    sum_df = pd.DataFrame(ensemble_event_num, columns=['sum'])
    result_array = pd.concat([event_header1, event_df, sum_df], axis=1)
    result_array.to_csv(ensemble_file, header=True, index=False)


def main():
    print("###########################################################")
    print("### ***            ########################################")
    print("###########################################################")

    event_file1 = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X22Y09\csv\MODIS_ndvi_mowing.csv'
    event_file2 = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X22Y09\csv\MODIS_evi_mowing.csv'
    ensemble_file = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X22Y09\csv\MODIS_mowing.csv'

    ensemble_detected_events_voting(event_file1, event_file2, ensemble_file)

    print("### Task over #############################################")


if __name__ == "__main__":
    main()
