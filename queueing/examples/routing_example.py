"""
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, bisect, brentq, minimize_scalar

import erlang.erlang_queues as eA


def agent_to_server(agents, hd):
    # converts agents counts to servers
    # hd: hours per day the agent works
    # work hours available in the queuing model = 24 * m
    # agents: total unique agents in a day
    # agents * hd: total work hours in a day
    # agents * hd / 24: work hours per hour = servers. Assumes a shift >= 1 hour
    return hd * agents / 24      # servers


def merge_types(a_df):
    # merge rows function
    d = dict()
    d['type'] = ['Trip']
    d['region'] = [a_df.loc[a_df.index[0], 'region']]
    d['language'] = [a_df.loc[a_df.index[0], 'language']]
    d['calls'] = [a_df['calls'].sum()]
    d['pct_abn'] = [(a_df['calls'] * a_df['pct_abn']).sum() / d['calls'][0]]
    d['p80'] = [(a_df['calls'] * a_df['p80']).sum() / d['calls'][0]]
    d['sla_120'] = [(a_df['calls'] * a_df['sla_120']).sum() / d['calls'][0]]
    return pd.DataFrame(d)


def get_aht(row, aht_tr, aht_nt):
    # return AHT based on type (TR, NT)
    if row['type'] == 'Trip':
        return aht_tr
    elif row['type'] == 'NonTrip':
        return aht_nt
    else:
        return None


def t_obj(p_tr, *args):
    # basic objective function for optimal routing: minimize the sum of mse of SLA and ABN for both trip and not trip
    # we could weight TR more heavily
    # Note that that obj function ensure to be close to the SLA and ABN but we may violate them
    # should try constraints on TR and minimize on NT
    calls, p_esc, t, calls_tr, calls_nt, tta_tr, tta_nt, aht_tr, aht_nt, m_tr, m_nt = args
    nt_abn = eA.abn_prob((1 - p_tr) * calls, 1 / aht_nt, m_nt, 1 / tta_nt)
    tr_abn = eA.abn_prob((p_tr + (1 - p_tr) * p_esc) * calls, 1 / aht_tr, m_tr, 1 / tta_tr)
    nt_sla = eA.sla_prob((1 - p_tr) * calls, 1 / aht_nt, m_nt, 1 / tta_nt, t)
    tr_sla = eA.sla_prob((p_tr + p_esc * (1 - p_tr)) * calls, 1 / aht_tr, m_tr, 1 / tta_tr, t)

    try:
        nt_abn_err = (0.15 - nt_abn) ** 2
        tr_abn_err = (0.15 - tr_abn) ** 2
        nt_sla_err = (0.8 - nt_sla) ** 2
        tr_sla_err = (0.8 - tr_sla) ** 2
        return tr_sla_err + tr_abn_err + nt_sla_err + nt_abn_err
    except TypeError:
        print('OBJ:::' + str(calls) + ' ' + str(p_tr) + ' nt: ' + str((1 - p_tr) * calls) + ' tr: ' + str((p_tr + p_esc * (1 - p_tr)) * calls,))
        return 100


def threshold(a_df, p_esc, t):
    # gets the optimal routing between TR and NT per language, region pair
    tr_row = a_df[a_df['type'] == 'Trip']
    nt_row = a_df[a_df['type'] == 'NonTrip']

    m_tr = tr_row['servers'].values[0]
    m_nt = nt_row['servers'].values[0]

    calls_tr = tr_row['calls'].values[0]
    calls_nt = nt_row['calls'].values[0]
    calls = calls_tr + calls_nt

    aht_tr = tr_row['mdl_aht'].values[0]
    aht_nt = nt_row['mdl_aht'].values[0]

    tta_tr = tr_row['mdl_tta'].values[0]
    tta_nt = nt_row['mdl_tta'].values[0]
    args = (calls, p_esc, t, calls_tr, calls_nt, tta_tr, tta_nt, aht_tr, aht_nt, m_tr, m_nt)
    p0 = (0.5, )
    res = minimize(t_obj, x0=p0, bounds=((0.1, 0.95), ), args=args, method='SLSQP')
    opt_pr = res.x[0]
    a_df.sort_values(by='type', inplace=True)  # NonTrip, Trip
    a_df['p_tr'] = [calls_nt / calls, calls_tr / calls]
    a_df['mdl_p_tr'] = [1 - opt_pr, opt_pr]
    a_df['mdl_adj_sla'] = [
        eA.sla_prob((1 - opt_pr) * calls, 1 / aht_nt, m_nt, 1 / tta_nt, t),
        eA.sla_prob((opt_pr + p_esc * (1 - opt_pr)) * calls, 1 / aht_tr, m_tr, 1 / tta_tr, t)
    ]
    a_df['mdl_adj_abn'] = [
        eA.abn_prob((1 - opt_pr) * calls, 1 / aht_nt, m_nt, 1 / tta_nt),
        eA.abn_prob((opt_pr + (1 - opt_pr) * p_esc) * calls, 1 / aht_tr, m_tr, 1 / tta_tr)
    ]
    return a_df


def fit_mdl_obj(x, *args):
    # obj function to fit TTA and AHT for each language, region, type combination to match ABN and SLA
    # note the obj is the generalized MAPE for each. This is to compensate over-weighting for low prob entries
    tta, aht = x
    row, t = args
    mdl_abn, abn = eA.abn_prob(row['calls'], 1 / aht, row['servers'], 1 / tta), row['pct_abn']
    mdl_sla, sla = eA.sla_prob(row['calls'], 1 / tta, row['servers'], 1 / aht, t), row['sla_120']
    return ((mdl_abn - abn) / (mdl_abn + abn)) ** 2 + ((mdl_sla - sla) / (mdl_sla + sla)) ** 2


def fit_mdl(row, aht_tr, aht_nt, t):
    # finds TTA and AHT for each row in the data
    # x0 = TTA, AHT
    aht = get_aht(row, aht_tr, aht_nt)
    bounds = ((1, 50), (0.5 * aht, 2.0 * aht))
    x0 = tuple([(x[0] + x[1]) / 2 for x in bounds])
    args = (row, t)
    res = minimize(fit_mdl_obj, x0=x0, bounds=bounds, args=args, method='SLSQP')
    if res.status == 0:
        row['mdl_tta'], row['mdl_aht'] = res.x
        row['mdl_pct_abn'] = eA.abn_prob(row['calls'], 1 / row['mdl_aht'], row['servers'], 1 / row['mdl_tta'])
        row['mdl_sla_120'] = eA.sla_prob(row['calls'], 1 / row['mdl_aht'], row['servers'], 1 / row['mdl_tta'], t)
    else:
        print('no convergence for ' + str(row))
    return row


# ###############################


AHT_tr = 17   # minutes
AHT_nt = 14   # minutes
hd = 8        # daily hours worked by an agent
p_esc = 0.1   # prob escalating to trip
t = 2         # p80 SLA time in mins

# data from dashboard
# need to fix NA before laoding, otherwise Pandas assumes NA is NaN
# rationalize names and units
lw_file = '~/Downloads/last_week.csv'
lw_data = pd.read_csv(lw_file)
lw_data.drop('sa_p95', axis=1, inplace=True)
lw_data.columns = ['type', 'region', 'language', 'calls', 'pct_abn', 'p80', 'sla_120']
lw_data['region'].replace('NOAM', 'NA', inplace=True)
lw_data['language'].replace('English-NA', 'English', inplace=True)
lw_data['language'].replace('English-EU', 'English', inplace=True)
lw_data['language'].replace('English-APAC', 'English', inplace=True)
lw_nt = lw_data[lw_data['type'] == 'NonTrip'].copy()
lw_ts = lw_data[lw_data['type'] != 'NonTrip'].copy()
lw_tr = lw_ts.groupby(['region', 'language']).apply(merge_types).reset_index(drop=True)
lw_data = pd.concat([lw_nt, lw_tr])
lw_data['calls'] /= (7 * 24 * 60)         # calls per minute

# agent counts
# need to fix NA before laoding, otherwise Pandas assumes NA is NaN
# rationalize names and units
a_file = '~/Downloads/agent_volume.csv'
ag_df = pd.read_csv(a_file)
ag_df = ag_df[(ag_df.ds >= '2017-12-05')]
ag_df.columns = ['ds', 'type', 'language', 'region', 'agents']
ag_df = ag_df[ag_df['language'].isin(lw_data['language'])]
ag_data = ag_df.groupby(['type', 'language', 'region']).agg({'agents': np.mean}).reset_index()   # avg unique daily agents
ag_data.replace('US', 'NA', inplace=True)
ag_data.replace('Europe', 'EMEA', inplace=True)

# baseline teh model
data = lw_data.merge(ag_data, on=['type', 'language', 'region'], how='inner')
data['servers'] = data['agents'].apply(lambda x: agent_to_server(x, hd))
mdl_data = data.apply(fit_mdl, axis=1, aht_tr=AHT_tr, aht_nt=AHT_nt, t=t)
mdl_data.plot(kind='scatter', x='pct_abn', y='mdl_pct_abn', title='ABN', grid=True)
mdl_data.plot(kind='scatter', x='sla_120', y='mdl_sla_120', title='SLA', grid=True)


# adjust routing threshold
data_adj = mdl_data.groupby(['language', 'region']).apply(threshold, p_esc=p_esc, t=t)

# data to scale up calls from Nov to Dec
calls_file = '~/Downloads/call_volume.csv'
calls_df = pd.read_csv(calls_file)
calls_df = calls_df[calls_df['ds'] >= '2017-12-06']
calls_df.columns = ['ds', 'language', 'region', 'type', 'calls']
calls_df = calls_df[(calls_df.language.notnull())]
calls_df['region'].replace('NOAM', 'NA', inplace=True)
calls_df['region'].replace('AP', 'APAC', inplace=True)
calls_df['region'].replace('EU', 'EMEA', inplace=True)
calls_df['language'].replace('CNMandarin', 'Mandarin', inplace=True)
calls_df = calls_df[calls_df['language'].isin(data['language'])]



