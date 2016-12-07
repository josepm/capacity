"""
compute system capacity from system throughput (X) and system response time (R)
detect capacity violations
detect capacity changes (drops mainly)

Overload Model
==============
- X tput (tps)
- R is avg apt (secs)
- T = XR  (traffic). Make sure units are consistent.
- P(m, T) = X^m/R (power), as a function of T with m >= 1 as a parameter.

Power
-----
- Power increases with T up to a max value Pmax(m) = P(m, Tmax) after that point it starts decreasing.
- Pmax is such that, (R'/R) = m * (X'/X) with X' = dX/dT and R' = dR/dT
- For m = 1,
  - Pmax(1) = P(1, Tmax) corresponds to Tmax(m=1) such that (R'/R) = (X'/X)
  - Before Pmax(1), if X'/X = x, then R'/R < x.
  - After Pmax(1), if X'/X = x, then R'/R > x.
  - After Tmax(m=1) contention starts
  - We can easily derive R(Tmax) and X(Tmax)
  - The system capacity is characterized by (Tmax, X(Tmax), R(Tmax))
- For m > 1,
  - Pmax(m) = P(m, Tmax) corresponds to Tmax(m) such that (R'/R) = m (X'/X).
  - For m > 1, we allow proportionally higher response time increases to achieve higher throughput.
  - Tmax(m) >= Tmax(1) which implies higher throughput and response time at Tmax(m) than at Tmax(1).
  - After Tmax(1) contention starts
  - If m > 1, we increase capacity (X) at the expense higher response R (due to contention). An increase

For M/M/1,
- T = rho = lambda / mu
- P(m, T) = T^m * (1 - T). This is off by a svc time factor from the general definition, which does not matter
- the maximimum is at T = m / (1 + m) which gives the max traffic

Algorithm  m >= 1.
=========
1) Compute the power as a function of X and R to have a DF with columns X, R, T = XR and P = X/R
1) compute quantiles for T and derive the corresponding quantile values for the power, X and R. Ensure ar least 20 or more points per quantile range.
2) Fit the power points into a function of the form a t^m + b t^(m+1). Check that a > 0 and b < 0
3) The maximum traffic is Tmax = (-a / b) * (m / (m+1))
4) Derive X(Tmax) and R(Tmax) from T = X*R and P = X^m/R for Pmax=P(m, Tmax)

import sys
import os
import numpy as np
import pandas as pd
import json
import statsmodels.api as sm

from Utilities import plotting_utils as pt_ut
from Utilities import stats_utils as st_ut
from Utilities import time_utils as tm_ut


def get_quantiles(df, t_col, qtiles):
    # returned a df with quantiles based on t_col quantiles
    qidx = t_col + '_qidx'
    q_df = pd.concat([df, pd.qcut(df[t_col], qtiles, labels=range(qtiles))], axis=1)            # t_col quantiles
    q_df.columns = list(df.columns) + [qidx]
    num_cols = [c for c in df.columns if str(df.dtypes[c]) != 'object']                         # only numerical cols
    g_df = q_df.groupby(qidx).agg({col: np.median for col in num_cols}).reset_index(drop=True)  # values in each quantile
    return g_df

def tput_engine_cap(a_df, x_col, y_col, t_col, qtiles, xexp=1.0, qsize=20, date=None, p_name=None):
    """
    finds capacity based on dR/R (response) and dX/X (demand)
    :param a_df: DF with data
    :param x_col: X values (input)
    :param y_col: R values (apt)
    :param t_col: T values (traffic)
    :param qtiles: number of quantiles to use
    :param date: end date of the data window
    :param last_result: output from the previous knee_der call
    :param qsize: min points in a quantile
    :param xexp: overload threshold. Normally 1. To increase capacity, set xexp to more than 1.
                  If xexp > 1, it detects a point past the optimal operating point and finds a higher demand level at higher response time.
    :return: [x_value, y_value, t_value]
    """
    k_df = a_df[[x_col, y_col, t_col]].copy()  # x_col = X, y_col = R, t_col = T
    k_df.sort_values(by=t_col, inplace=True)
    k_df.reset_index(inplace=True, drop=True)
    k_df['pwr'] = (1000.0 * k_df[x_col] / 3600.0) ** xexp/ (k_df[y_col] / 1000.0)

    g_df = get_quantiles(k_df, t_col, qtiles) if len(k_df) >= qtiles * qsize else k_df   # use quantiles if enough points, otherwise raw data

    # pwr = a * t + b * t^(2*m)
    y = g_df['pwr'].values
    X = np.array(pd.concat([g_df[t_col] ** xexp, g_df[t_col] ** (1 + xexp)], axis=1))
    model = sm.OLS(y, X)
    results = model.fit()
    a, b = results.params
    if a > 0.0 and b < 0.0:
        q_rsq_adj = results.rsquared_adj
        t_val = (-a / b) * (xexp / (1.0 + xexp))
        pwr_val = a * np.power(t_val, xexp) + b * np.power(t_val, 1.0 + xexp)
        x_val = np.power(t_val * pwr_val, 1.0 / (1.0 + xexp)) * (3600.0 / 1000.0)            # Kusers/hr
        y_val = np.power(np.power(t_val, xexp) / pwr_val, 1.0 / (1.0 + xexp)) * 1000.0       # msecs
        return [x_val, y_val, t_val, q_rsq_adj]
    else:
        print 'invalid fit on date ' + str(date) + ' for ' + str(p_name)
        return [np.nan, np.nan, np.nan, np.nan]
        

def cap_defects(data_df, cap_df, d_col, time_fmt, defect_window=7):
    """
    compute daily defect rate
    :param in_df: DF with data from all pods
    :param p_name: pod name
    :param cap_df: capacity DF for p_name only
    :param apt_max: max APT (to trim in_df)
    :param defect_window: number of days used to compute the defect rate
    :param d_col: defect column name
    :return:
    """
    # data_df = in_df[(in_df['pod'] == p_name) & (in_df['apt'] <= apt_max)].copy()  # get pod data
    data_df['day'] = data_df['time'].apply(tm_ut.change_format, in_format=time_fmt, out_format='%Y-%m-%d')

    # knee using c_reqs qtiles
    start_date = cap_df['day'].min()
    end_date = cap_df['day'].max()
    end_window = tm_ut.add_days(start_date, defect_window, date_format='%Y-%m-%d')
    w_df_list = list()
    while start_date <= end_date:
        w_df = data_df[(start_date <= data_df['time']) & (data_df['time'] < end_window)].copy()
        df = w_df.merge(cap_df, on='day', how='left')
        df[d_col + '_defect'] = df.apply(lambda x: 1 if x[d_col] > x[d_col + '_cap'] else 0, axis=1)
        df['cnt'] = 1
        w_df = df.groupby(['pod', 'day']).agg({d_col + '_defect': np.sum, 'cnt': np.sum, 'tps_cap': np.mean, 'apt_cap': np.mean}).reset_index()
        w_df[d_col + '_defect'] /= w_df['cnt']
        w_df_list.append(w_df[['pod', 'day', d_col + '_defect', 'tps_cap', 'apt_cap']])
        start_date = tm_ut.add_days(start_date, defect_window, date_format='%Y-%m-%d')
        end_window = tm_ut.add_days(start_date, defect_window, date_format='%Y-%m-%d')
    defect_df = pd.concat(w_df_list)
    defect_df.reset_index(inplace=True, drop=True)
    return defect_df

