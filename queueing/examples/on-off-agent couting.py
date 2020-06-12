"""

"""

from capacity_planning.queueing import erlang as ERL

def get_agents(wh, hrs, hbat, theta, q_sla, t_sla, util, shkg, mdl, wweek=40):
    calls = 60 * wh / hbat  # calls
    lbda = calls / (hrs * 60)  # arrivals (calls per minute)

    ss = ERL.ServerSizing(mdl,
                          {'lbda': lbda, 'mu': mu, 'theta': theta},
                          cons={'sla': ['sla_func', q_sla, t_sla], 'util': ['util_func', util]}, verbose=False)
    m = ss.num_servers(util)
    return ss.get_agents(m, shkg, hrs, wweek)


# ############################
HBAT = 30.0      # minutes/call
mu = 1 / HBAT   # svc rate
TTA = 2         # minutes
theta = 1 / TTA
t_SLA = 2       # minutes
q_SLA = 0.8    # SLA quantile
max_util = 0.8   # occupancy
shrinkage = 0.25   # agent shrinkage
# #############################

model = 'ErlangA'
ttl_wh = 100000                           # whours
ttl_hours = 24 * 7                        # ttl hours
peak_wh = ttl_wh * 0.75                # wh received during peak
peak_hours = ttl_hours * 0.25             # hours at peak
off_peak_wh = ttl_wh - peak_wh            # wh received off peak
off_peak_hours = ttl_hours - peak_hours   # hours off peak
languages = [0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]

flat_agents = get_agents(ttl_wh, ttl_hours, HBAT, theta, q_SLA, t_SLA, max_util, shrinkage, model, wweek=40)                    # Flat model
peak_agents = get_agents(peak_wh, peak_hours, HBAT, theta, q_SLA, t_SLA, max_util, shrinkage, model, wweek=40)                  # peak model
off_peak_agents = get_agents(off_peak_wh, off_peak_hours, HBAT, theta, q_SLA, t_SLA, max_util, shrinkage, model, wweek=40)      # off peak model

print('flat_agents: ' + str(flat_agents) + ' peak: ' + str(peak_agents) + ' off_peak: ' + str(off_peak_agents) + ' all: ' + str(peak_agents + off_peak_agents))


# try by language
flat_agents, peak_agents, off_peak_agents = 0, 0, 0
for p in languages:
    wh = ttl_wh * p
    peak_wh = wh * 0.75              # wh received during peak
    off_peak_wh = wh - peak_wh  # wh received off peak
    flat_agents = get_agents(wh, ttl_hours, HBAT, theta, q_SLA, t_SLA, max_util, shrinkage, model, wweek=40)                    # Flat model
    peak_agents = get_agents(peak_wh, peak_hours, HBAT, theta, q_SLA, t_SLA, max_util, shrinkage, model, wweek=40)                  # peak model
    off_peak_agents = get_agents(off_peak_wh, off_peak_hours, HBAT, theta, q_SLA, t_SLA, max_util, shrinkage, model, wweek=40)      # off peak model
    print('prob: ' + str(p) + ' flat_agents: ' + str(flat_agents) + ' peak: ' + str(peak_agents) + ' off_peak: ' + str(off_peak_agents) + ' all: ' + str(peak_agents + off_peak_agents))