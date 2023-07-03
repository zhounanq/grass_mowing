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


################# user defined parameters #################
global g_season_start, g_season_end, g_end_grass_season, g_peak_start, g_peak_end, g_gaussian_std, g_number_positive_eval, g_min_dayinterval

# 定义草地生长季的开始与结束
# define the approximate length of grassland season in which you expect the main mowing activity; in decimal years = DOY / 365; make sure too include a temporal buffer --> here end of December
g_season_start = 0.2  # DOY 73
g_season_end = 1  # DOY 365

# define end of grassland season
g_end_grass_season = 0.85  # DOY

# 定义草地最旺盛的时间范围(比如6月到7月)
# define the approximate length of the main vegetation season; i.e., time of the year in which you expect at least one peak
g_peak_start = 0.33  # DOY 120
g_peak_end = 0.66  # DOY 240

# adjust sensitivity of thresholds; i.e., width of gaussian function and number of positive evaluations needed
g_gaussian_std = 0.02
g_number_positive_eval = 40

# 定义两次割草时间之间的最小间隔
# define minimum distance between two consecutive mowing events in days
g_min_dayinterval = 15
###########################################################


def sg_filter_stastic(xs, ys, nodata=-9999):

    if np.any(xs == nodata):
        return -1, -1

    # s-g filtering
    filted_xs = signal.savgol_filter(xs, 21, 3)

    min_value = np.min(filted_xs)
    min_index = np.argmin(filted_xs)

    max_value = np.max(filted_xs)
    max_index = np.argmax(filted_xs)

    print("MIN:[{}, {}]; MAX:[{}, {}]".format(min_index, min_value, max_index, max_value))
    return min_index, max_index


def adjust_timeseries_min(xs, ys, min_index):

    std_xs = np.roll(xs, -min_index)
    std_ys = np.roll(ys, -min_index)

    return std_xs, std_ys


def whole_min_value_index(xs, ys):

    csv_shape = xs.shape
    (row, col) = csv_shape
    print("")

    min_index_array = np.zeros(row)
    max_index_array = np.zeros(row)

    for rr in range(0, row):

        row_array = xs.iloc[rr:rr+1,:].to_numpy()
        row_array = row_array.astype('int')*0.0001

        min_idx, max_idx = sg_filter_stastic(row_array, ys)
        min_index_array[rr] = min_idx
        max_index_array[rr] = max_idx
    # for

    count_dict = {}
    for i in set(min_index_array):
        count_dict[i] = list(min_index_array).count(i)
    count_dict = sorted(count_dict.items(), key=lambda item: (-item[1], item[0]))
    print(count_dict)

    pass


def check_time_series_array(ts_array, nodata=-9999):

    nan_num = np.isnan(ts_array).sum()
    nodata_num = (ts_array==nodata).sum()
    nan_num_ratio = nan_num / len(ts_array)
    nodata_num_ratio = nodata_num / len(ts_array)

    if (nan_num_ratio > 0.5) or (nodata_num_ratio > 0.5) :
        return False

    return True


def stastic_time_series_array(x, y, nodata=-9999, season_start=0.2, season_end=0.85):
    """

    :param x:
    :param y:
    :param nodata:
    :param season_start:
    :param season_end:
    :return: [ratio of no_data], [max gap in array],
    """
    if np.all(y == nodata):
        return 0, (x[-1] - x[0])*365

    ###########################################################
    # ratio of no_data
    nodata_sum = np.sum(np.where(y==nodata, True, False))
    nodata_ratio = 1-(nodata_sum/len(y))

    ###########################################################
    data_gap = 0
    data_gap_indeces = []
    data_gap_dates_list = []
    for index, value in enumerate(y):
        if value == nodata:
            if index < 1:
                continue
            data_gap += 1
            if data_gap == 0:
                data_gap_indeces.append(index-1)
            data_gap_indeces.append(index)
        else:
            if len(x[data_gap_indeces]) >= 1:
                data_gap_indeces.append(index)
                gap_dates = x[data_gap_indeces]
                gap_days = (gap_dates[-1]-gap_dates[0])*365
                data_gap_dates_list.append(gap_days)
            else:
                data_gap_dates_list.append(0)
            data_gap = 0
            data_gap_indeces = []
        # else
    # for

    ###########################################################
    # calculating gap to EOS
    index_to_end_save = -1
    for indeces_to_end in range(1, len(y)):
        if y[-indeces_to_end] == nodata:
            index_to_end_save = -(indeces_to_end + 1)
            continue
        else:
            break
    gap_to_end = (season_end - x[index_to_end_save]) * 365
    data_gap_dates_list.append(gap_to_end)

    ###########################################################
    # calculating gap to SOS
    index_to_start_save = 0
    for indeces_to_start in range(len(y)):
        if y[indeces_to_start] == nodata:
            index_to_start_save = (indeces_to_start + 1)
            continue
        else:
            break
    gap_to_start = (x[index_to_start_save]-season_start) * 365
    data_gap_dates_list.append(gap_to_start)

    # if no gap is found it will return 5 days as gap
    # in case the last potential observation misses
    # the function calculates the gap to the EOS
    if int(max(data_gap_dates_list)) == 0:
        data_gap_dates_list.append(5)

    return nodata_ratio, max(data_gap_dates_list)


def detect_mowing_event(xs, ys, min_interval, type='ConHull', nOrder=3, model='linear'):
    """

    xs: 草地时间序列特征数组
    ys: 草地时间序列节点数组
    min_interval: 两次割草时间之间的最小间隔
    """

    # Mowing events followed by unnaturally quick regrowth larger than
    # a user defined fixed threshold (here: 0.15) within 5 days, are excluded.
    # This pattern is likely caused by an undetected cloud or cloud shadow.
    quick_regrow_threshold = 0.15

    Y = np.asarray(ys)
    X = np.asarray(xs)

    # 本年度的草地生长季开始与结束
    season_start_frac = g_season_start
    season_end_frac = g_season_end

    # 本年度的最旺盛时间范围
    peak_start_frac = g_peak_start
    peak_end_frac = g_peak_end
    
    # Xarr, Yarr = search_key_points(xs, ys, min_interval, type)

    if type == 'ConHull':

        # 寻找最接近生长季开始与结束的数据时间点,提取生长季期间内的数据
        season_start_diff = np.abs(Y - season_start_frac)
        obs_start_index = np.nanargmin(season_start_diff)
        season_end_diff = np.abs(Y - season_end_frac)
        obs_end_index = np.nanargmin(season_end_diff)
        Y = Y[obs_start_index:obs_end_index]
        X = X[obs_start_index:obs_end_index]

        # calculate VI difference (t1) - (t-1)
        grow_diff_array = np.append([0], np.diff(X))

        # calculate statistic values of time-series VI
        season_length = int((Y[-1] - Y[0]) * 365)
        season_vi_std = np.nanstd(X)
        season_vi_mean = np.nanmean(X)
        available_vi_obs = sum(~np.isnan(X))
        available_vi_obs_ratio1 = available_vi_obs / len(X)
        available_vi_obs_ratio2 = available_vi_obs / (season_length / 5)


        ###############################################################
        # identify first peak somewhere around the "mid" of the season
        mid_start_diff = np.abs(Y - peak_start_frac)
        mid_start_index = np.nanargmin(mid_start_diff)
        mid_end_diff = np.abs(Y - peak_end_frac)
        mid_end_index = np.nanargmin(mid_end_diff)

        vi_peak_sub = X[mid_start_index : mid_end_index]
        try: # raise exception
            if len(vi_peak_sub) == 0:
                raise ValueError
        except Exception as e:
            print("")

        if np.all(np.isnan(vi_peak_sub)):
            return None, None
        season_mid_index = np.nanargmax(vi_peak_sub)
        season_mid_index = mid_start_index + season_mid_index


        ###############################################################
        early_peak_index_2 = []
        late_peak_index_2 = []
        # search the second biggest VI in the first-half season and the second-half season
        if season_mid_index <= 2:
            early_peak_vi_1 = np.nanmax(X[0 : season_mid_index])
            early_peak_index_1 = np.min(np.where(X == early_peak_vi_1))
        else:
            search_ind_1 = np.argwhere(Y <= Y[season_mid_index] - min_interval * 0.00273973)
            if np.any(search_ind_1):
                search_ind_1 = search_ind_1.max()
                early_peak_vi_1 = np.nanmax(X[0:search_ind_1])
                if np.any(~np.isnan(early_peak_vi_1)):
                    # maybe values in early_peak_vi_1 equal to np.nan
                    early_peak_index_1 = np.min(np.where(X == early_peak_vi_1))
                else:
                    early_peak_index_1 = 0
            else:
                early_peak_index_1 = 0
        # if

        if (early_peak_index_1 != 0) and (early_peak_index_1 - 2) > 0 and np.any(X[0:early_peak_index_1 - 2]):
            search_ind_2 = np.argwhere(Y <= Y[early_peak_index_1] - min_interval * 0.00273973)
            if np.any(search_ind_2):
                search_ind_2 = search_ind_2.max()
                early_peak_vi_2 = np.nanmax(X[0:search_ind_2])
                if np.any(~np.isnan(early_peak_vi_2)):
                    # maybe values in early_peak_vi_2 equal to np.nan
                    early_peak_index_2 = np.min(np.where(X == early_peak_vi_2))
        # if

        # in the second-half season
        if season_mid_index + 2 == len(X):
            late_peak_vi_1 = np.nanmax(X[season_mid_index + 1:len(X)])
            late_peak_index_1 = np.max(np.where(X == late_peak_vi_1))
        else:
            search_ind_3 = np.argwhere(Y >= Y[season_mid_index] + min_interval * 0.00273973)
            if np.any(search_ind_3):
                search_ind_3 = search_ind_3.min()
                if search_ind_3 != len(Y)-1:
                    late_peak_vi_1 = np.nanmax(X[search_ind_3:len(Y)-1])
                    if np.any(~np.isnan(late_peak_vi_1)):
                        # maybe values in early_peak_vi_1 equal to np.nan
                        late_peak_index_1 = np.max(np.where(X == late_peak_vi_1))
                    else:
                        late_peak_index_1 = 0
                else:
                    late_peak_index_1 = 0
            else:
                late_peak_index_1 = 0

        if (late_peak_index_1 != 0) and late_peak_index_1 + 2 <= len(X) and np.any(X[late_peak_index_1 + 2:len(X)]):
            search_ind_4 = np.argwhere(Y >= Y[late_peak_index_1] + min_interval * 0.00273973)
            if np.any(search_ind_4):
                search_ind_4 = search_ind_4.min()
                late_peak_vi_2 = np.nanmax(X[search_ind_4:len(Y)])
                if np.any(~np.isnan(late_peak_vi_2)):
                    # maybe values in early_peak_vi_2 equal to np.nan
                    late_peak_index_2 = np.max(np.where(X == late_peak_vi_2))

        ###############################################################
        # todo check if early and late peak equals Y0
        Y0 = np.argwhere(np.isfinite(Y))
        Y0 = np.min(Y0)

        Xarr = [X[Y0], X[early_peak_index_1], X[season_mid_index], X[late_peak_index_1], X[len(X) - 1]]
        Yarr = [Y[Y0], Y[early_peak_index_1], Y[season_mid_index], Y[late_peak_index_1], Y[len(Y) - 1]]
        if early_peak_index_2:
            Xarr = [X[Y0], X[early_peak_index_2], X[early_peak_index_1], X[season_mid_index], X[late_peak_index_1], X[len(X) - 1]]
            Yarr = [Y[Y0], Y[early_peak_index_2], Y[early_peak_index_1], Y[season_mid_index], Y[late_peak_index_1], Y[len(Y) - 1]]
            if late_peak_index_2:
                Xarr = [X[Y0], X[early_peak_index_2], X[early_peak_index_1], X[season_mid_index], X[late_peak_index_1], X[late_peak_index_2], X[len(X) - 1]]
                Yarr = [Y[Y0], Y[early_peak_index_2], Y[early_peak_index_1], Y[season_mid_index], Y[late_peak_index_1], Y[late_peak_index_2], Y[len(Y) - 1]]
        if late_peak_index_2:
            Xarr = [X[Y0], X[early_peak_index_1], X[season_mid_index], X[late_peak_index_1], X[late_peak_index_2], X[len(X) - 1]]
            Yarr = [Y[Y0], Y[early_peak_index_1], Y[season_mid_index], Y[late_peak_index_1], Y[late_peak_index_2], Y[len(Y) - 1]]
            if early_peak_index_2:
                Xarr = [X[Y0], X[early_peak_index_2], X[early_peak_index_1], X[season_mid_index], X[late_peak_index_1], X[late_peak_index_2], X[len(X) - 1]]
                Yarr = [Y[Y0], Y[early_peak_index_2], Y[early_peak_index_1], Y[season_mid_index], Y[late_peak_index_1], Y[late_peak_index_2], Y[len(Y) - 1]]
    else:
        print("")
        return None, None

    ###############################################################
    if model == 'linear':
        fit_val = np.interp(Y, xp=Yarr, fp=Xarr)

    if model == 'poly':
        # model and fit poly of n-th order
        poly = np.polyfit(Yarr, Xarr, nOrder)
        fit_val = np.polyval(poly, Y)

    if model == 'spline':
        tck = interpolate.splrep(x=Yarr, y=Xarr, s=0)
        fit_val = interpolate.splev(Y, tck, der=0)


    ###############################################################
    # difference between polynom and values
    diff = np.abs(fit_val - X)
    diff_sum = np.nansum(diff)
    diff_std = np.nanstd(diff)
    diff_mean = np.nanmean(diff)
    diff_med = np.nanmedian(diff)

    # calculate the dynamic thresholds
    thresh = diff_mean + diff_std * 2.5
    vi_threshold = -season_vi_std
    vi_threshold_list = list(np.random.normal(vi_threshold, g_gaussian_std, 100))


    ###############################################################
    mowing_obs_list = [] # index of mowing in time-series observation
    mowing_doy_list = [] # index of mowing in DOY

    if len(diff) > 0:
        i = 1
        for diff_index, diff_value in enumerate(diff):

            vi_check_list = [grow_diff_array[diff_index]] * 100
            check_result = [a for a, b in zip(vi_check_list, vi_threshold_list) if a < b]
            if len(check_result) < g_number_positive_eval:
                continue

            check_doy = Y[diff_index]
            if diff_index == len(Y)-1:
                check_doy_next = Y[diff_index] + 1
            else:
                check_doy_next = Y[diff_index + 1]

            if i == 1:
                # search the first mowing event
                if diff_value > thresh:
                    # Mowing events followed by unnaturally quick regrowth larger than
                    # a user defined fixed threshold (here: 0.15) within 5 days, are excluded.
                    # This pattern is likely caused by an undetected cloud or cloud shadow.
                    if (check_doy_next - check_doy <= 6 * 0.00273973) and (grow_diff_array[diff_index + 1] > quick_regrow_threshold):
                        continue
                    # get julian date
                    doy = ((check_doy) * 365) + 1
                    if doy <= (g_end_grass_season * 365 - 5):
                        mowing_doy_list.append(int(doy))
                        mowing_obs_list.append(diff_index)
                        i = i + 1
                    else:
                        # 草地生长季结束前5天内，不会发生割草事件
                        continue
                    # if
                # if
            else:
                # search the non-first mowing event
                if diff_value > thresh:
                    # Potential mowing events must be more than 15 days apart from each other,
                    # as this depicts real-world regrowth potential and management opportunities.
                    dec_date_preceding = Y[np.array(mowing_obs_list)[-1]]
                    dec_date_current_iter = Y[diff_index]
                    if (dec_date_current_iter - dec_date_preceding) > (min_interval / 365):
                        # 短时间内的急剧升高(小于6天的快速生长)可能是异常噪声
                        if (check_doy_next - check_doy <= 6 * 0.00273973) and (grow_diff_array[diff_index + 1] > quick_regrow_threshold):
                            continue
                        # get julian date
                        doy = ((check_doy) * 365) + 1
                        if doy <= (g_end_grass_season * 365 - 5):
                            # Between two potential mowing events, there must at least be one
                            # observation with a value higher than the preceding one (i.e., a positive ΔY),
                            # as there must be regrowth to justify another mowing event.
                            # check if there is one observation that is higher than the preceding between two mowing events
                            time_mask = np.where((Y >= Y[mowing_obs_list[-1]]) & (Y <= check_doy), True, False)
                            any_preced_lower = np.any(np.ediff1d(X[time_mask]) > 0)
                            # in case there is no increase in EVI values between two mowing events "any_preced_lower" will be False
                            if any_preced_lower:
                                mowing_doy_list.append(int(doy))
                                mowing_obs_list.append(diff_index)
                                i = i + 1
                        else:
                            continue
                    else:
                        # when day interval is smaller than 15 days, we pass this event.
                        None
                # if
            # if
        # for
    # if

    return mowing_doy_list, diff_sum


def test_modis_evi():

    # cmd line
    vi_array = np.array([5217.48,3768.85,4953.30,4248.96,4726.06,4360.70,3731.27,3906.97,
                            4748.92,4334.91,4013.00,4660.40,5550.28,4812.33,4724.10,4592.64,
                            4767.46,4157.37,3652.08,3696.46,3964.88,3271.59,3791.03,2742.15,
                            2786.12,2694.34,2792.41,2549.41,2413.17,2195.35,2279.46,2107.55,
                            1975.94,2135.46,2236.30,2668.38,2639.51,2814.47,2469.68,2895.35,
                            3919.37,2755.27,3540.38,4239.37,3991.34,4542.21])
    tstep_array = np.array([1,9,17,25,33,41,49,57,65,73,81,89,97,
                            105,113,121,129,137,145,153,161,169,177,185,193,
                            201,209,217,225,233,241,249,257,265,273,281,289,297,
                            305,313,321,329,337,345,353,361])

    # detecting
    nodata_ratio, max_gap_days = stastic_time_series_array(tstep_array/365.0, vi_array)
    mowing_doy, diff_sum = detect_mowing_event(vi_array, tstep_array/365.0, min_interval=15,
                                               type='ConHull', nOrder=2, model='poly')
    print(nodata_ratio, max_gap_days, mowing_doy)

    # drawing picture
    fig = plt.figure(figsize=(10, 6))
    plt.plot(tstep_array, vi_array, linestyle='-', marker='*', c='r', alpha=0.5)
    for doy in mowing_doy:
        plt.vlines(doy, 0, 1, colors="b", linestyles="dashed")

    plt.xlabel('date', fontsize=14)
    plt.ylabel("VI", fontsize=16)
    plt.show()


def test_modis_ndvi():

    bandnames = ['mowingEvents', 'max_gap_days', 'CSO_ABS', 'Data_Ratio',
                 'Mow_1', 'Mow_2', 'Mow_3', 'Mow_4', 'Mow_5', 'Mow_6', 'Mow_7', 'Mean', 'Median', 'SD', 'diff_sum',
                 'diff_sum_dataavail', 'Error']

    # cmd line
    index_array = np.array([0.30,0.30,0.35,0.30,0.30,0.20,0.25,0.40,0.65,0.82,0.45,0.64,0.68,
                            0.72,0.84,0.60,0.77,0.80,0.40,0.50,0.60,0.40,0.40,0.50,0.25])
    tstep_array = np.array([0,15,30,45,60,75,90,105,120,135,150,165,180,
                            195,210,225,240,255,270,285,300,315,330,345,360])

    # detecting
    nodata_ratio, max_gap_days = stastic_time_series_array(tstep_array/365.0, index_array)
    mowing_doy, diff_sum = detect_mowing_event(index_array, tstep_array/365.0, min_interval=15,
                                               type='ConHull', nOrder=2, model='poly')
    print(nodata_ratio, max_gap_days, mowing_doy)

    # drawing picture
    fig = plt.figure(figsize=(10, 6))
    plt.plot(tstep_array, index_array, linestyle='-', marker='*', c='r', alpha=0.5)
    for doy in mowing_doy:
        plt.vlines(doy, 0, 1, colors="b", linestyles="dashed")

    plt.xlabel('date', fontsize=14)
    plt.ylabel("VI", fontsize=16)
    plt.show()


def main_XXXYYYY(in_vicsv, out_mowcsv, min_idx):
    print("###########################################################")
    print("### Grass mowing event detection ##########################")
    print("###########################################################")

    tstep_array_modis = np.array([1,9,17,25,33,41,49,57,65,73,81,89,97,
                                  105,113,121,129,137,145,153,161,169,177,185,193,
                                  201,209,217,225,233,241,249,257,265,273,281,289,297,
                                  305,313,321,329,337,345,353,361])

    vicsv_file = in_vicsv
    result_file = out_mowcsv

    ###########################################################
    # hyper parameters for X22Y07
    no_data = -0.9999
    minval_idx = min_idx
    max_mowing = 9
    global g_season_start; g_season_start = 0.1
    global g_season_end; g_season_end = 0.9
    # define end of grassland season
    global g_end_grass_season; g_end_grass_season = 0.85  # DOY
    global g_peak_start; g_peak_start = 0.33  # DOY 120
    global g_peak_end; g_peak_end = 0.66  # DOY 240

    ###########################################################
    # reading csv data
    csv_data = pd.read_csv(vicsv_file, header=0)

    csv_header = csv_data.iloc[:, :5]
    csv_array = csv_data.iloc[:, 5:]

    (row, col) = csv_array.shape
    print("### Total row {}, and total time step {}".format(row, col))

    ###########################################################
    # detecting mowing events
    mowing_doy_array = np.zeros([row, max_mowing], dtype='int')
    for rr in range(0, row):

        row_array = np.array(csv_array.iloc[rr]).astype('int')
        if not check_time_series_array(row_array):
            print("ROW {}: Time-series data wrong".format(rr))
            continue

        row_array = row_array*0.0001
        row_array[row_array==no_data] = np.nan

        re_xs, re_ys = adjust_timeseries_min(row_array, tstep_array_modis, min_index=minval_idx)

        mowing_doy, diff_sum = detect_mowing_event(re_xs, tstep_array_modis / 365.0, min_interval=15, type='ConHull', nOrder=2, model='poly')
        if mowing_doy is None or diff_sum is None:
            print("ROW {}: Nothing found".format(rr))
            continue

        mowing_doy = np.array([(doy+8*minval_idx)%368 for doy in mowing_doy]) - 1
        mowing_doy = np.sort(mowing_doy)
        if len(mowing_doy) > max_mowing:
            mowing_doy = mowing_doy[:max_mowing]
        mowing_doy = np.pad(mowing_doy, (0, max_mowing-len(mowing_doy)), 'constant', constant_values=(0, 0))
        mowing_num = sum(mowing_doy != 0)

        mowing_doy_array[rr] = mowing_doy
        print("ROW {}: Mowing events @ {}".format(rr, mowing_doy))
    # for

    ###########################################################
    # writing results
    array_columns = ["mow_{}".format(ii+1) for ii in range(0, max_mowing)]
    result_array = pd.DataFrame(mowing_doy_array, columns=array_columns)
    result_df = pd.concat([csv_header, result_array], axis=1)
    result_df.to_csv(result_file, header=True, index=False)

    print("### Task over #############################################")


def main_X21Y07(in_vicsv, out_mowcsv):
    print("###########################################################")
    print("### Grass mowing event detection ##########################")
    print("###########################################################")

    tstep_array_modis = np.array([1,9,17,25,33,41,49,57,65,73,81,89,97,
                                  105,113,121,129,137,145,153,161,169,177,185,193,
                                  201,209,217,225,233,241,249,257,265,273,281,289,297,
                                  305,313,321,329,337,345,353,361])

    vicsv_file = in_vicsv
    result_file = out_mowcsv

    ###########################################################
    # hyper parameters for X21Y07
    no_data = -0.9999
    minval_idx = 30
    max_mowing = 9
    global g_season_start; g_season_start = 0.1
    global g_season_end; g_season_end = 0.9
    # define end of grassland season
    global g_end_grass_season; g_end_grass_season = 0.85  # DOY
    global g_peak_start; g_peak_start = 0.33  # DOY 120
    global g_peak_end; g_peak_end = 0.66  # DOY 240

    ###########################################################
    # reading csv data
    csv_data = pd.read_csv(vicsv_file, header=0)

    csv_header = csv_data.iloc[:, :5]
    csv_array = csv_data.iloc[:, 5:]

    (row, col) = csv_array.shape
    print("### Total row {}, and total time step {}".format(row, col))

    ###########################################################
    # detecting mowing events
    mowing_doy_array = np.zeros([row, max_mowing], dtype='int')
    for rr in range(0, row):

        row_array = csv_array.iloc[rr].to_numpy().astype('int')
        if not check_time_series_array(row_array):
            print("ROW {}: Time-series data wrong".format(rr))
            continue

        row_array = row_array*0.0001
        row_array[row_array==no_data] = np.nan

        re_xs, re_ys = adjust_timeseries_min(row_array, tstep_array_modis, min_index=minval_idx)

        mowing_doy, diff_sum = detect_mowing_event(re_xs, tstep_array_modis / 365.0, min_interval=15,
                                                   type='ConHull', nOrder=2, model='poly')

        mowing_doy = np.array([(doy+8*minval_idx)%368 for doy in mowing_doy]) - 1
        mowing_doy = np.sort(mowing_doy)
        if len(mowing_doy) > max_mowing:
            mowing_doy = mowing_doy[:max_mowing]
        mowing_doy = np.pad(mowing_doy, (0, max_mowing-len(mowing_doy)), 'constant', constant_values=(0, 0))
        mowing_num = sum(mowing_doy != 0)

        mowing_doy_array[rr] = mowing_doy
        print("ROW {}: Mowing events @ {}".format(rr, mowing_doy))
    # for

    ###########################################################
    # writing results
    array_columns = ["mow_{}".format(ii+1) for ii in range(0,9)]
    result_array = pd.DataFrame(mowing_doy_array, columns=array_columns)
    result_array = pd.concat([csv_header, result_array], axis=1)
    result_array.to_csv(result_file, header=True, index=False)

    print("### Task over #############################################")


def main_X22Y07(in_vicsv, out_mowcsv):
    print("###########################################################")
    print("### Grass mowing event detection ##########################")
    print("###########################################################")

    tstep_array_modis = np.array([1,9,17,25,33,41,49,57,65,73,81,89,97,
                                  105,113,121,129,137,145,153,161,169,177,185,193,
                                  201,209,217,225,233,241,249,257,265,273,281,289,297,
                                  305,313,321,329,337,345,353,361])

    vicsv_file = in_vicsv
    result_file = out_mowcsv

    ###########################################################
    # hyper parameters for X22Y07
    no_data = -0.9999
    minval_idx = 9
    max_mowing = 9
    global g_season_start; g_season_start = 0.1
    global g_season_end; g_season_end = 0.9
    # define end of grassland season
    global g_end_grass_season; g_end_grass_season = 0.85  # DOY
    global g_peak_start; g_peak_start = 0.33  # DOY 120
    global g_peak_end; g_peak_end = 0.66  # DOY 240

    ###########################################################
    # reading csv data
    csv_data = pd.read_csv(vicsv_file, header=0)

    csv_header = csv_data.iloc[:, :5]
    csv_array = csv_data.iloc[:, 5:]

    (row, col) = csv_array.shape
    print("### Total row {}, and total time step {}".format(row, col))

    ###########################################################
    # detecting mowing events
    mowing_doy_array = np.zeros([row, max_mowing], dtype='int')
    for rr in range(0, row):

        row_array = np.array(csv_array.iloc[rr]).astype('int')
        if not check_time_series_array(row_array):
            print("ROW {}: Time-series data wrong".format(rr))
            continue

        row_array = row_array*0.0001
        row_array[row_array==no_data] = np.nan

        re_xs, re_ys = adjust_timeseries_min(row_array, tstep_array_modis, min_index=minval_idx)

        mowing_doy, diff_sum = detect_mowing_event(re_xs, tstep_array_modis / 365.0, min_interval=15, type='ConHull', nOrder=2, model='poly')
        if mowing_doy is None or diff_sum is None:
            print("ROW {}: Nothing found".format(rr))
            continue

        mowing_doy = np.array([(doy+8*minval_idx)%368 for doy in mowing_doy]) - 1
        mowing_doy = np.sort(mowing_doy)
        if len(mowing_doy) > max_mowing:
            mowing_doy = mowing_doy[:max_mowing]
        mowing_doy = np.pad(mowing_doy, (0, max_mowing-len(mowing_doy)), 'constant', constant_values=(0, 0))
        mowing_num = sum(mowing_doy != 0)

        mowing_doy_array[rr] = mowing_doy
        print("ROW {}: Mowing events @ {}".format(rr, mowing_doy))
    # for

    ###########################################################
    # writing results
    array_columns = ["mow_{}".format(ii+1) for ii in range(0, max_mowing)]
    result_array = pd.DataFrame(mowing_doy_array, columns=array_columns)
    result_df = pd.concat([csv_header, result_array], axis=1)
    result_df.to_csv(result_file, header=True, index=False)

    print("### Task over #############################################")


def main_X22Y08(in_vicsv, out_mowcsv):
    print("###########################################################")
    print("### Grass mowing event detection ##########################")
    print("###########################################################")

    tstep_array_modis = np.array([1,9,17,25,33,41,49,57,65,73,81,89,97,
                                  105,113,121,129,137,145,153,161,169,177,185,193,
                                  201,209,217,225,233,241,249,257,265,273,281,289,297,
                                  305,313,321,329,337,345,353,361])

    vicsv_file = in_vicsv
    result_file = out_mowcsv

    ###########################################################
    # hyper parameters for X22Y08
    no_data = -0.9999
    minval_idx = 17
    max_mowing = 9
    global g_season_start; g_season_start = 0.1
    global g_season_end; g_season_end = 0.9
    # define end of grassland season
    global g_end_grass_season; g_end_grass_season = 0.85  # DOY
    global g_peak_start; g_peak_start = 0.33  # DOY 120
    global g_peak_end; g_peak_end = 0.66  # DOY 240

    ###########################################################
    # reading csv data
    csv_data = pd.read_csv(vicsv_file, header=0)

    csv_header = csv_data.iloc[:, :5]
    csv_array = csv_data.iloc[:, 5:]

    (row, col) = csv_array.shape
    print("### Total row {}, and total time step {}".format(row, col))

    ###########################################################
    # detecting mowing events
    mowing_doy_array = np.zeros([row, max_mowing], dtype='int')
    for rr in range(0, row):

        row_array = np.array(csv_array.iloc[rr]).astype('int')
        if not check_time_series_array(row_array):
            print("ROW {}: Time-series data wrong".format(rr))
            continue

        row_array = row_array*0.0001
        row_array[row_array==no_data] = np.nan

        re_xs, re_ys = adjust_timeseries_min(row_array, tstep_array_modis, min_index=minval_idx)

        mowing_doy, diff_sum = detect_mowing_event(re_xs, tstep_array_modis / 365.0, min_interval=15, type='ConHull', nOrder=2, model='poly')
        if mowing_doy is None or diff_sum is None:
            print("ROW {}: Nothing found".format(rr))
            continue

        mowing_doy = np.array([(doy+8*minval_idx)%368 for doy in mowing_doy]) - 1
        mowing_doy = np.sort(mowing_doy)
        if len(mowing_doy) > max_mowing:
            mowing_doy = mowing_doy[:max_mowing]
        mowing_doy = np.pad(mowing_doy, (0, max_mowing-len(mowing_doy)), 'constant', constant_values=(0, 0))
        mowing_num = sum(mowing_doy != 0)

        mowing_doy_array[rr] = mowing_doy
        print("ROW {}: Mowing events @ {}".format(rr, mowing_doy))
    # for

    ###########################################################
    # writing results
    array_columns = ["mow_{}".format(ii+1) for ii in range(0, max_mowing)]
    result_array = pd.DataFrame(mowing_doy_array, columns=array_columns)
    result_df = pd.concat([csv_header, result_array], axis=1)
    result_df.to_csv(result_file, header=True, index=False)

    print("### Task over #############################################")


def main_X22Y09(in_vicsv, out_mowcsv):
    print("###########################################################")
    print("### Grass mowing event detection ##########################")
    print("###########################################################")

    tstep_array_modis = np.array([1,9,17,25,33,41,49,57,65,73,81,89,97,
                                  105,113,121,129,137,145,153,161,169,177,185,193,
                                  201,209,217,225,233,241,249,257,265,273,281,289,297,
                                  305,313,321,329,337,345,353,361])

    vicsv_file = in_vicsv
    result_file = out_mowcsv

    ###########################################################
    # hyper parameters for X22Y09
    no_data = -0.9999
    minval_idx = 18
    max_mowing = 9
    global g_season_start; g_season_start = 0.1
    global g_season_end; g_season_end = 0.9
    # define end of grassland season
    global g_end_grass_season; g_end_grass_season = 0.85  # DOY
    global g_peak_start; g_peak_start = 0.33  # DOY 120
    global g_peak_end; g_peak_end = 0.66  # DOY 240

    ###########################################################
    # reading csv data
    csv_data = pd.read_csv(vicsv_file, header=0)

    csv_header = csv_data.iloc[:, :5]
    csv_array = csv_data.iloc[:, 5:]

    (row, col) = csv_array.shape
    print("### Total row {}, and total time step {}".format(row, col))

    ###########################################################
    # detecting mowing events
    mowing_doy_array = np.zeros([row, max_mowing], dtype='int')
    for rr in range(0, row):

        row_array = np.array(csv_array.iloc[rr]).astype('int')
        if not check_time_series_array(row_array):
            print("ROW {}: Time-series data wrong".format(rr))
            continue

        row_array = row_array*0.0001
        row_array[row_array==no_data] = np.nan

        re_xs, re_ys = adjust_timeseries_min(row_array, tstep_array_modis, min_index=minval_idx)

        mowing_doy, diff_sum = detect_mowing_event(re_xs, tstep_array_modis / 365.0, min_interval=15, type='ConHull', nOrder=2, model='poly')
        if mowing_doy is None or diff_sum is None:
            print("ROW {}: Nothing found".format(rr))
            continue

        mowing_doy = np.array([(doy+8*minval_idx)%368 for doy in mowing_doy]) - 1
        mowing_doy = np.sort(mowing_doy)
        if len(mowing_doy) > max_mowing:
            mowing_doy = mowing_doy[:max_mowing]
        mowing_doy = np.pad(mowing_doy, (0, max_mowing-len(mowing_doy)), 'constant', constant_values=(0, 0))
        mowing_num = sum(mowing_doy != 0)

        mowing_doy_array[rr] = mowing_doy
        print("ROW {}: Mowing events @ {}".format(rr, mowing_doy))
    # for

    ###########################################################
    # writing results
    array_columns = ["mow_{}".format(ii+1) for ii in range(0, max_mowing)]
    result_array = pd.DataFrame(mowing_doy_array, columns=array_columns)
    result_df = pd.concat([csv_header, result_array], axis=1)
    result_df.to_csv(result_file, header=True, index=False)

    print("### Task over #############################################")


def main_X23Y06(in_vicsv, out_mowcsv):
    print("###########################################################")
    print("### Grass mowing event detection ##########################")
    print("###########################################################")

    tstep_array_modis = np.array([1,9,17,25,33,41,49,57,65,73,81,89,97,
                                  105,113,121,129,137,145,153,161,169,177,185,193,
                                  201,209,217,225,233,241,249,257,265,273,281,289,297,
                                  305,313,321,329,337,345,353,361])

    vicsv_file = in_vicsv
    result_file = out_mowcsv

    ###########################################################
    # hyper parameters for X23Y06
    no_data = -0.9999
    minval_idx = 30
    max_mowing = 9
    global g_season_start; g_season_start = 0.1
    global g_season_end; g_season_end = 0.9
    # define end of grassland season
    global g_end_grass_season; g_end_grass_season = 0.85  # DOY
    global g_peak_start; g_peak_start = 0.33  # DOY 120
    global g_peak_end; g_peak_end = 0.66  # DOY 240

    ###########################################################
    # reading csv data
    csv_data = pd.read_csv(vicsv_file, header=0)

    csv_header = csv_data.iloc[:, :5]
    csv_array = csv_data.iloc[:, 5:]

    (row, col) = csv_array.shape
    print("### Total row {}, and total time step {}".format(row, col))

    ###########################################################
    # detecting mowing events
    mowing_doy_array = np.zeros([row, max_mowing], dtype='int')
    for rr in range(0, row):

        row_array = csv_array.iloc[rr].to_numpy().astype('int')
        if not check_time_series_array(row_array):
            print("ROW {}: Time-series data wrong".format(rr))
            continue

        row_array = row_array*0.0001
        row_array[row_array==no_data] = np.nan

        re_xs, re_ys = adjust_timeseries_min(row_array, tstep_array_modis, min_index=minval_idx)

        mowing_doy, diff_sum = detect_mowing_event(re_xs, tstep_array_modis / 365.0, min_interval=15,
                                                   type='ConHull', nOrder=2, model='poly')

        mowing_doy = np.array([(doy+8*minval_idx)%368 for doy in mowing_doy]) - 1
        mowing_doy = np.sort(mowing_doy)
        if len(mowing_doy) > max_mowing:
            mowing_doy = mowing_doy[:max_mowing]
        mowing_doy = np.pad(mowing_doy, (0, max_mowing-len(mowing_doy)), 'constant', constant_values=(0, 0))
        mowing_num = sum(mowing_doy != 0)

        mowing_doy_array[rr] = mowing_doy
        print("ROW {}: Mowing events @ {}".format(rr, mowing_doy))
    # for

    ###########################################################
    # writing results
    array_columns = ["mow_{}".format(ii+1) for ii in range(0, max_mowing)]
    result_array = pd.DataFrame(mowing_doy_array, columns=array_columns)
    result_df = pd.concat([csv_header, result_array], axis=1)
    result_df.to_csv(result_file, header=True, index=False)

    print("### Task over #############################################")


def main():
    print("###########################################################")
    print("### Grass mowing event detection ##########################")
    print("###########################################################")

    # in_vicsv = r'I:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X21Y07\csv\csv_vits_MODIS_ndvi_c.csv'
    # out_mowcsv = r'I:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X21Y07\csv\csv_vits_MODIS_ndvi_c_mowing.csv'
    # main_X21Y07(in_vicsv, out_mowcsv)

    in_vicsv = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X22Y09\csv\MODIS_ndvi.csv'
    out_mowcsv = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X22Y09\csv\MODIS_ndvi_mowing.csv'
    main_X22Y09(in_vicsv, out_mowcsv)

    # in_vicsv = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X23Y06\csv\X23Y06_MODIS_evi.csv'
    # out_mowcsv = r'J:\FF\application_dataset\africa_grass\01-Probav\PROBAV_S10_TOC_R1k\PROBAV_S10_TOC_X23Y06\csv\X23Y06_MODIS_evi_mowing.csv'
    # main_X23Y06(in_vicsv, out_mowcsv)

    print("### Task over #############################################")


if __name__ == "__main__":
    main()
