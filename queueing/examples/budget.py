import os
import sys
import numpy as np
import pandas as pd

import erlang.erlang_queues as eA

# ############### Global parameters


AHT = 17.0      # minutes
q_SLA = 0.8     # p(Wait > t) <= 0.8

# derived
mu = 1 / AHT  # svc rate

# ######################################### Budget example


def sla_view(gf):
    w = gf[['Agents', 'prob_abn', 'Agent_util']].copy().T
    w.columns = gf['SLA_time'].values
    return w


lbda_arr, TTA_arr, m_arr, pab_arr, t_arr, q_arr, util_arr = list(), list(), list(), list(), list(), list(), list()
for TTA in [5, 10, 15]:
    theta = 1 / TTA
    for lbda in np.linspace(10, 100, 10):
        m_max = 4 * lbda / mu
        for t in np.linspace(0, 5, 6):
            m = m_max
            w = eA.sla_prob(lbda, mu, m_max, theta, t) - q_SLA
            while w >= 0:
                m -= 1
                w = eA.sla_prob(lbda, mu, m, theta, t) - q_SLA

            pab = eA.abn_prob(lbda, mu, m, theta)
            lbda_arr.append(lbda)
            TTA_arr.append(TTA)
            m_arr.append(m)
            pab_arr.append(pab)
            t_arr.append(t)
            q_arr.append(eA.sla_prob(lbda, mu, m, theta, t))
            util_arr.append(lbda * (1 - pab) / (m * mu))
            print('lbda: ' + str(lbda) + ' t: ' + str(t) + ' m: ' + str(int(m)) + ' pab: ' + str(eA.abn_prob(lbda, mu, m, theta)) + ' q: ' + str(eA.sla_prob(lbda, mu, m, theta, t)))
df = pd.DataFrame({'interactions/min': lbda_arr, 'prob_abn': pab_arr, 'TTA': TTA_arr, 'Agents': m_arr, 'SLA_time': t_arr, 'SLA_q': q_arr, 'Agent_util': util_arr})


zf = df.groupby(['interactions/min', 'TTA']).apply(sla_view).reset_index()
zf.columns = ['interactions/min', 'TTA', 'metric', 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

agents_df = zf[zf.metric == 'Agents'].copy()
pabn_df = zf[zf.metric == 'prob_abn'].copy()
putil_df = zf[zf.metric == 'Agent_util'].copy()

agents_df.drop('metric', axis=1, inplace=True)
pabn_df.drop('metric', axis=1, inplace=True)
putil_df.drop('metric', axis=1, inplace=True)

agents_df.columns = ['interactions/min', 'TTA', 'Agents::0mins', 'Agents::1mins','Agents::2mins','Agents::3mins','Agents::4mins', 'Agents::5mins']
y_df = agents_df.set_index(['interactions/min', 'TTA'])
rel_agents_df = y_df.apply(lambda x: x / x['Agents::2mins'], axis=1).reset_index()

pabn_df.columns = ['interactions/min', 'TTA', 'ABN_prob::0mins', 'ABN_prob::1mins','ABN_prob::2mins','ABN_prob::3mins','ABN_prob::4mins', 'ABN_prob::5mins']
putil_df.columns = ['interactions/min', 'TTA', 'Agent_util::0mins', 'Agent_util::1mins','Agent_util::2mins','Agent_util::3mins','Agent_util::4mins', 'Agent_util::5mins']

df1 = rel_agents_df.merge(pabn_df, on=['interactions/min', 'TTA'], how='inner')
d_df = df1.merge(putil_df, on=['interactions/min', 'TTA'], how='inner')

d_df.to_csv('/tmp/budget.csv', index=False)
