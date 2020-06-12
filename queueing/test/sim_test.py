"""
simulator tests
"""

from capacity_planning.queueing import erlang_tools as et
from capacity_planning.queueing import solver as solver
from capacity_planning.queueing import erlang_queues as eq

# ################# ERLANG C #########################
tba = 0.022132235578917376
tis = 11.16955118472165
mdl_pars = {'lbda': 1/tba, 'mu': 1/tis}
mdl ='ErlangC'

# one constraint
cons = {'sla': {'sla_func': [2, 0.85]}}
ss = et.ServerSizing(mdl, mdl_pars, cons=cons)
m_c1 = ss.get_servers()                            # number of servers to meet SLA
ec = eq.ErlangC(1.0 / tba, 1.0 / tis, m_c1)
pw_c1 = ec.sla_prob(2)                             # prob(wait <= 2) should be >= 85%
util_c1 = ec.utilization()

# simulation
s_cfg = {
    "reuse": False, "verbose": True, "num_customers": 500, "servers": None,
    "sla_obj": {
        "max_abn_prob": 0.05,
        "wait_thres": 2,
        "sla_q": 0.85,
    },
    "tba": {'dist_name': 'expon', 'params': [[1.0], [tba]], 'avg': tba, 'std': tba},
    "tis": {'dist_name': 'expon', 'params': [[1.0], [tis]],'avg': tis, 'std': tis},
    "tta": None
}
sim_stats_df_c1, res_dict_c1 = solver.solver(s_cfg)    # number of servers should less than 2% from the one from the computation

# two constraints
cons = {'sla': {'sla_func': [2, 0.85]}, 'util': {'util_func': [0.80]}}
ss = et.ServerSizing(mdl, mdl_pars, cons=cons)
m_c2 = ss.get_servers()                            # number of servers to meet SLA
ec = eq.ErlangC(1.0 / tba, 1.0 / tis, m_c2)
pw_c2 = ec.sla_prob(2)                             # prob(wait <= 2) should be >= 85%
util_c2 = ec.utilization()                         # utilization should be <= 80%
s_cfg = {
    "reuse": False, "verbose": True, "num_customers": 500, "servers": None,
    "sla_obj": {
        "max_abn_prob": 0.05,
        "wait_thres": 2,
        "sla_q": 0.85,
        "max_util": 0.80
    },
    "tba": {'dist_name': 'expon', 'params': [[1.0], [tba]], 'avg': tba, 'std': tba},
    "tis": {'dist_name': 'expon', 'params': [[1.0], [tis]],'avg': tis, 'std': tis},
    "tta": None
}
sim_stats_df_c2, res_dict_c2 = solver.solver(s_cfg)    # number of servers should about 2% from the one from the computation

# ################# ERLANG A #########################
tta = tis / 2                             # patience equal to 0.5 svc time
mdl_pars = {'lbda': 1/tba, 'mu': 1/tis, 'theta': 1/tta}
mdl = 'ErlangA'

# one constraint
cons = {'sla': {'sla_func': [2, 0.85]}}
ss = et.ServerSizing(mdl, mdl_pars, cons=cons)
m_a1 = ss.get_servers()                            # number of servers to meet SLA
ea = eq.ErlangA(1.0 / tba, 1.0 / tis, m_a1, 1.0 / tta)
pw_a1 = ea.sla_prob(2)                             # prob(wait <= 2) should be >= 85%
util_a1 = ea.utilization()                         # utilization should be <= 80%
s_cfg = {
    "reuse": False, "verbose": True, "num_customers": 500, "servers": None,
    "sla_obj": {
        "wait_thres": 2,
        "sla_q": 0.85,
    },
    "tba": {'dist_name': 'expon', 'params': [[1.0], [tba]], 'avg': tba, 'std': tba},
    "tis": {'dist_name': 'expon', 'params': [[1.0], [tis]],'avg': tis, 'std': tis},
    "tta": {'dist_name': 'expon', 'params': [[1.0], [tta]],'avg': tta, 'std': tta}
}
sim_stats_df_a1, res_dict_a1 = solver.solver(s_cfg)    # number of servers should less than 2% from the one from the computation

# two constraints
cons = {'sla': {'sla_func': [2, 0.85]}, 'abn': {'abn_func': [0.05]}}
ss = et.ServerSizing(mdl, mdl_pars, cons=cons)
m_a2 = ss.get_servers()                            # number of servers to meet SLA
ea = eq.ErlangA(1.0 / tba, 1.0 / tis, m_a2, 1.0 / tta)
pw_a2 = ea.sla_prob(2)                             # prob(wait <= 2) should be >= 85%
util_a2 = ea.utilization()                         # utilization should be <= 80%
abn_a2 = ea.abn_prob()
s_cfg = {
    "reuse": False, "verbose": True, "num_customers": 500, "servers": None,
    "sla_obj": {
        "max_abn_prob": 0.05,
        "wait_thres": 2,
        "sla_q": 0.85,
    },
    "tba": {'dist_name': 'expon', 'params': [[1.0], [tba]], 'avg': tba, 'std': tba},
    "tis": {'dist_name': 'expon', 'params': [[1.0], [tis]],'avg': tis, 'std': tis},
    "tta": {'dist_name': 'expon', 'params': [[1.0], [tta]],'avg': tta, 'std': tta}
}
sim_stats_df_a2, res_dict_a2 = solver.solver(s_cfg)    # number of servers should less than 2% from the one from the computation
