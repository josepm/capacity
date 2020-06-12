"""
erlang examples
Usage: $ python examples.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from capacity_planning.queueing.erlang import erlang_tools as ERL
from capacity_planning.queueing.erlang import erlang_queues as EQ

# import machine_repair as MR

# ################## VOICE #########################
# ################## VOICE #########################
# ################## VOICE #########################
# ################## VOICE #########################
# ################## VOICE #########################

print('\n\n############## VOICE EXAMPLES #################')

# ##################################################
# ############### Global parameters ################
# ##################################################


lbda = 20.0       # arrivals (calls per minute)
AHT = 17.0      # minutes
TTA = 1.0        # minutes

# SLA: prob(Wait <= t_SLA) = 0.8
t_SLA = 1.0      # minutes
q_SLA = 0.95     # SLA quantile

# ABN prob(ABN) <= p_ABN
p_ABN = 0.02

occupancy = 1.0
shrinkage = 0.3   # agent shrinkage
shift_hrs = 8.0

# ############################## cost minimization
h_wage = 20      # hourly wage
ltv = 1000       # customer LTV
a = 0.1          # cost (arbitrary) parameter

# ###############
mu = 1 / AHT    # svc rate
theta = 1 / TTA

ss = ERL.ServerSizing('ErlangA', {'lbda': lbda, 'mu': mu, 'theta': theta},
                      cons={
                          'sla': {'sla_func': [q_SLA, t_SLA]},
                          'abn': {'abn_func': [p_ABN]},
                          'util': {'util_func': [1.0]}
                      }, verbose=False)
m = ss.get_servers()
# agents = ss.server_to_agent(shrinkage, occupancy, shift_hrs)
print('servers: ' + str(m))

eq = EQ.ErlangC(lbda, mu, 353)
prob_wait = eq.erlC()
avg_wait = eq.avg_wait()
util = eq.utilization()
sys_avg = eq.sys_avg()
print('prob wait: ' + str(prob_wait) + ' avg_wait: ' + str(avg_wait) + ' util: ' + str(util) + ' sys_avg: ' + str(sys_avg))


def cost(x, args):  # cost per call
    _lbda, _mu, _theta, _t, _ltv, _a, _hw = args
    return lbda * (1 - ERL.ErlangA(_lbda, _mu, x[0], _theta).sla_prob(_t)) * _ltv * _a + x[0] * _hw / 60


ss = ERL.ServerSizing('ErlangA', {'lbda': lbda, 'mu': mu, 'theta': theta},
                      cons={'sla': {'sla_func': [q_SLA, t_SLA]}, 'abn': {'abn_func': [p_ABN]}}, verbose=False)
m = ss.get_servers()
agents = ss.server_to_agent(shrinkage, servers=m)

erla = EQ.ErlangA(lbda, mu, m, theta,  verbose=False)
q = erla.sla_prob(t_SLA)
res = minimize_scalar(lambda x: (q_SLA - erla.sla_prob(x)) ** 2, bounds=(0, 2 * t_SLA), method='bounded')
t0 = res.x if res.status == 0 else 0
u = erla.utilization()
print('Constrained Cost minimization::\tservers: ' + str(m) + ' agents: ' + str(agents))
print('\tutilization target: ' + str(max_util) + '  achieved utilization: ' + str(u))
print('\tp80 SLA target: ' + str(t_SLA) + '  achieved p80 SLA: ' + str(res.x))

# plot
m_vals = np.array(range(1, 1000))
y_vals = np.array([ERL.ErlangA(lbda, mu, m, 1 / TTA, verbose=False).sla_prob(t_SLA) for m in m_vals])
z_vals = np.array([ERL.ErlangA(lbda, mu, m, 1 / TTA, verbose=False).abn_prob() for m in m_vals])
u_vals = np.array([ERL.ErlangA(lbda, mu, m, 1 / TTA, verbose=False).utilization() for m in m_vals])
c_vals = np.array([cost([m], [lbda, mu, theta, t_SLA, ltv, a, h_wage]) for m in m_vals])

df = pd.DataFrame({'Servers': m_vals, 'Prob(W <= T & !ABN)': y_vals, 'Prob(ABN)': z_vals, 'Agent_Util': u_vals, 'Cost': c_vals})
df.set_index('Servers', inplace=True)

# non-blocking plot
plt.ion()
plt.show()
ax = df.plot(title='Cost Based Call Center Sizing', grid=True, secondary_y=['Cost'])
ax.set_xlabel('Servers')
ax.set_ylabel('Probability')
ax.right_ax.set_ylabel('Cost')
plt.show()
plt.draw()
plt.pause(0.001)
input("Press [enter] to continue.")

# ############################# Pool splitting
trip = 0.3
non_trip = 1 - trip
esc = 0.05
lbda_peak = 50

print('\nPool Splitting')
# One pool of agents
ss = ERL.ServerSizing('ErlangA', {'lbda': lbda, 'mu': mu, 'theta': theta}, cons={'sla': ['sla_func', q_SLA, t_SLA]}, verbose=False)
m = ss.get_servers(lambda x: x)
print('Agents with one pool: ' + str(m))

# Two pools of agent pools
ss = ERL.ServerSizing('ErlangA', {'lbda': non_trip * lbda, 'mu': mu, 'theta': theta}, cons={'sla': ['sla_func', q_SLA, t_SLA]}, verbose=False)
m_nt = ss.get_servers(lambda x: x)
ss = ERL.ServerSizing('ErlangA', {'lbda': (trip + esc) * lbda, 'mu': mu, 'theta': theta}, cons={'sla': ['sla_func', q_SLA, t_SLA]}, verbose=False)
m_tr = ss.get_servers(lambda x: x)
print('Total agents with two pools: ' + str(m_nt + m_tr) + ' trip: ' + str(m_tr) + ' non-trip: ' + str(m_nt))


# ############################## multiple SLAs
off_peak_months = 8
off_peak_lbda = 10
off_peak_t = t_SLA / 2

peak_months = 12 - off_peak_months
peak_lbda = (12 * lbda + off_peak_months * off_peak_lbda) / peak_months
peak_t = 2 * t_SLA - off_peak_t

ss = ERL.ServerSizing('ErlangA', {'lbda': off_peak_lbda, 'mu': mu, 'theta': theta}, cons={'sla': ['sla_func', q_SLA, off_peak_t]}, verbose=False)  # off_peak
m_off_peak = ss.get_servers(lambda x: x)
ss = ERL.ServerSizing('ErlangA', {'lbda': peak_lbda, 'mu': mu, 'theta': theta}, cons={'sla': ['sla_func', q_SLA, peak_t]}, verbose=False)  # peak
m_peak = ss.get_servers(lambda x: x)
m2 = (m_off_peak * off_peak_months + m_peak * peak_months) / 12
ss = ERL.ServerSizing('ErlangA', {'lbda': lbda, 'mu': mu, 'theta': theta}, cons={'sla': ['sla_func', q_SLA, t_SLA]}, verbose=False)
m1 = ss.get_servers(lambda x: x)

print('\nPeak/Off Peak SLAs')
print('\tpeak SLA p80:' + str(peak_t) + ' calls during peak: ' + str(peak_lbda) + ' agents during peak: ' + str(m_peak))
print('\toff-peak SLA p80:' + str(off_peak_t) + ' calls during off peak: ' + str(off_peak_lbda) + ' agents during off peak: ' + str(m_off_peak))
print('Single SLA')
print('\tpeak SLA p80:' + str(t_SLA) + ' avg calls: ' + str(lbda) + ' agents: ' + str(m1))
print('Summary\n\tavg agents with 2 SLAs: ' + str(int(m2)) + ' agents with 1 SLA: ' + str(m1))


# ############################# MESSAGING #################################
# ############################# MESSAGING #################################
# ############################# MESSAGING #################################
# ############################# MESSAGING #################################
# ############################# MESSAGING #################################
# ############################# MESSAGING #################################
# ############################# MESSAGING #################################

print('\n\n############## MESSAGING EXAMPLES #################')

# ##################################################
# ############### Global parameters ################
# ##################################################


lbda = 10       # arrivals (threads per minute)
AHT = 15.0      # minutes
mu = 1 / AHT    # svc rate for a whole thread
TTA = 10        # minutes
theta = 1 / TTA
max_util = 0.80  # agent occupancy
q = 0.8          # context switching cost

AIT = 3            # avg interaction time by the agent
mu_mr = 1 / AIT    # svc rate for a single interaction
TT = 5     # avg customer think time
lbda_mr = 1 / TT

q_SLA = 0.8
t_SLA = 2
i_SLA = 10         # avg time between interactions within a thread

# find the concurrency
for jobs in range(1, 10):
    mr = MR.MachineRepair(lbda_mr, mu_mr, 1, jobs)
    print('jobs: ' + str(jobs) + ' util: ' + str(1 - mr.probs[0]) + ' response time: ' + str(mr.cycle_time - 1 / lbda))

K = 2      # based on the previous print out
ss = ERL.ServerSizing('ErlangA_PS', {'lbda': lbda, 'mu': mu, 'theta': theta, 'q': q, 'K': K}, cons={'sla': ['sla_func', q_SLA, t_SLA]}, verbose=False)
m = ss.get_servers(lambda x: x)
print('Messaging System Sizing:: server concurrency: ' + str(K) + ' servers: ' + str(m))
