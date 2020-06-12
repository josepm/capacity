"""
solve Erlang through simulation
$ python solver.py config/solver_cfg.json
depending on cfg,
- if the server count is not provided, find optimal number of servers if they are not provided in cfg
- if servers provided, output SLA performance (response time, abn and utilization)
"""


import os
import json
import random
import numpy as np
from itertools import product

from capacity_planning.queueing import erlang_tools as e_tools
from capacity_planning.utilities import stats_utils as sut
from capacity_planning.utilities import sys_utils as s_ut

# parameters
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
FILE_PATH = os.path.dirname(os.path.abspath(__file__))


class DistroSim(object):
    def __init__(self, name, dist_name, params):
        self.mdl_name = name
        self.dist_name = dist_name
        self.distro_obj = sut.DistributionMixer(self.dist_name, params=params)
        self.params = params
        try:
            self.avg = self.distro_obj.mean()
        except TypeError:
            s_ut.my_print(str(os.getpid()) + ' WARNING: no mean for ' + self.mdl_name)
            self.avg = np.nan
        try:
            self.std = self.distro_obj.std()
        except TypeError:
            s_ut.my_print(str(os.getpid()) + ' WARNING: no std for ' + self.mdl_name)
            self.std = np.nan
        s_ut.my_print('pid: ' + str(os.getpid()) + ' DistroSim::' + str(self.__str__()))

    def rvs(self, size):
        return self.distro_obj.rvs(size=size)

    def __str__(self):
        return 'mdl name: ' + self.mdl_name + ' ' + self.distro_obj.__str__()


class SimSolver(object):
    def __init__(self, s_cfg):
        # cfg dict must have keys:
        # tba, tis, tta: all have format {'dist_name': <dist_name>, 'params': [probs, pars] where probs array with values that sum to 1 and pars from scipy.stat dist params
        # sla_obj,
        # reuse, num_customers, servers, verbose
        arr_mdl, svc_mdl, tta_mdl, sla_obj = s_cfg['tba'], s_cfg['tis'], s_cfg.get('tta', None), s_cfg['sla_obj']
        reuse, n_customers, servers, verbose, ts_key = s_cfg['reuse'], s_cfg['num_customers'], s_cfg['servers'], s_cfg['verbose'], s_cfg['ts_key']
        self.is_valid = True
        self.n_customers = n_customers   # customers per server (so that we have enough data
        self.reuse = reuse
        self.max_customers = 1000000
        self.m_mdl = None
        self.ts_key = ts_key
        self.do_sim = True

        self.arr_mdl = DistroSim('tba', arr_mdl['dist_name'], arr_mdl['params'])
        self.avg_arr = self.arr_mdl.avg
        self.std_arr = self.arr_mdl.std

        self.svc_mdl = DistroSim('tis', svc_mdl['dist_name'], svc_mdl['params'])
        self.avg_svc = self.svc_mdl.avg
        self.std_svc = self.svc_mdl.std

        self.mdl_dict = {
            'tba': {'dist_name': arr_mdl['dist_name'], 'params': arr_mdl['params']},
            'tis': {'dist_name': svc_mdl['dist_name'], 'params': svc_mdl['params']}
        }

        if np.isinf(self.avg_svc) or self.avg_svc <= 0 or np.isinf(self.std_svc) or self.std_svc <= 0:
            s_ut.my_print(str(os.getpid()) + ' invalid values: ' + str(self.svc_mdl.__str__()))
            self.is_valid = False

        if tta_mdl is not None:
            self.tta_mdl = DistroSim('tta', tta_mdl['dist_name'], tta_mdl['params'])
            self.mdl_dict['tta'] = {'dist_name': tta_mdl['dist_name'], 'params': tta_mdl['params']}
            self.avg_tta = self.tta_mdl.avg
            self.std_tta = self.tta_mdl.std
            if np.isinf(self.avg_tta) or self.avg_tta <= 0 or np.isinf(self.std_tta) or self.std_tta <= 0:
                s_ut.my_print(str(os.getpid()) + ' invalid values: ' + str(self.tta_mdl.__str__()))
                self.is_valid = False
        else:
            self.tta_mdl = None
            self.avg_tta = np.inf
            self.std_tta = np.nan

        # fill SLA with a no-op sla when missing
        if sla_obj is None:  # can be None if servers is not None. In this case default values are set
            sla_obj = dict()
        self.max_abn_prob = sla_obj.get('max_abn_prob', 1.0)
        self.wait_thres = sla_obj.get('wait_thres', 0.0)
        self.sla_q = sla_obj.get('sla_q', 0.0)
        self.max_util = sla_obj.get('max_util', 1.0)
        self.sla_dict = sla_obj

        self.idx_arr = None
        self.verbose = verbose
        self.run_results = dict()    # dict with key = nbr servers and values T/F (true means SLA met)
        self.servers = None if servers is None else int(np.floor(servers))   # model seems to over estimate
        self.ss_obj = None
        self.min_servers = 1         # adjust for ErlangC later if needed

        # get analytical server counts and avg in system
        self.lbda = 1.0 / self.avg_arr
        self.mu = 1.0 / self.avg_svc
        self.theta = 0.0 if self.tta_mdl is None else 1.0 / self.avg_tta
        if self.theta == 0.0:
            self.sla_dict = {'sla': {'sla_func': [self.wait_thres, self.sla_q]}}
        else:
            self.sla_dict = {'sla': {'sla_func': [self.wait_thres, self.sla_q]}, 'abn': {'abn_func': [self.max_abn_prob]}}
        s_ut.my_print('pid: ' + str(os.getpid()) + ' analytical mdl pars:: lbda: ' + str(self.lbda) + ' mu: ' + str(self.mu) + ' theta: ' + str(self.theta))

        if self.lbda <= 0.0 or self.mu <= 0.0 or self.theta < 0.0 or np.isnan(self.lbda) or np.isnan(self.mu) or np.isnan(self.theta):
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR_:get_emdl:params invalid::' + self.ss_obj.__str__())
            self.m_mdl, self.min_servers = None, None
            self.is_valid = False
        else:
            self.is_valid = True
            s_ut.my_print('pid: ' + str(os.getpid()) + ' mdl pars:: ' + str(self.mdl_dict))
            self.get_emdl()    # set analytical values
            s_ut.my_print('pid: ' + str(os.getpid()) + ' basic mdl:: lbda: ' + str(self.lbda) + ' mu: ' + str(self.mu) + ' theta: ' + str(self.theta) + ' m: ' + str(self.m_mdl))

    def get_emdl(self):    # set basic analytical server counts and avg in system
        self.ss_obj = self._set_emdl(self.lbda, self.mu, self.theta, self.sla_dict, self.verbose)
        self.min_servers = self.ss_obj.min_mval
        self.m_mdl = self.ss_obj.get_servers()
        if self.m_mdl is None:
            s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: analytical model failed')
        return

    @staticmethod
    def _set_emdl(lbda, mu, theta, sla_dict, verbose):    # set analytical server counts and avg in system
        mdl = 'ErlangC' if theta is None or theta == 0.0 else 'ErlangA'
        return e_tools.ServerSizing(mdl, {'lbda': lbda, 'mu': mu, 'theta': theta}, cons=sla_dict, verbose=verbose)

    def _get_erl_m(self, lbda, mu, theta, prob=1.0):    # get servers (m) for a partial model
        ss_obj = self._set_emdl(lbda, mu, theta, self.sla_dict, self.verbose)
        min_servers = ss_obj.min_mval
        m = ss_obj.get_servers()
        s_ut.my_print('pid: ' + str(os.getpid()) + ' get_erl::prob: ' + str(prob) + ' lbda: ' + str(lbda) + ' mu: ' + str(mu) + ' theta: ' + str(theta) + ' m: ' + str(m))
        if m is None:
            return None
        else:
            return max(m, min_servers)

    @staticmethod
    def _get_dcmdl(in_mdl, size, em_phases=1, max_splits=2):  # fit arr_v to em_phases of exponentials (partial models)
        arr_v = in_mdl.rvs(size=size)
        dc_em = sut.HyperExp(arr_v, em_phases=em_phases, max_splits=max_splits, floc=0.0)
        dc_em.fit()
        if dc_em.fit_res is None:
            return None
        else:
            v_out = np.array([(d['prob'], 1.0 / d['params'][-1]) for d in dc_em.fit_res])  # (prob, rate)
            if dc_em.m_err > 0.25 and dc_em.s_err > 0.5:
                v_out = np.array([(1.0, 1.0 / in_mdl.avg)])  # replace by plain exponential
                s_ut.my_print('pid: ' + str(os.getpid()) + ' Poor fit for input model: ' + in_mdl.__str__() + ' replacing by exponential: ' + str(v_out))
            return v_out, dc_em.em_mean, dc_em.em_std

    def get_dcmdl(self, size=1000):
        # divide and conquer model: break data into groups with CV <= max_CV and apply EM with em_phases. Each partial model is M/M/m.

        s_ut.my_print('pid: ' + str(os.getpid()) + ' *********** dc model for tba')
        arrs = self._get_dcmdl(self.arr_mdl, size, em_phases=2, max_splits=1)

        s_ut.my_print('pid: ' + str(os.getpid()) + ' *********** dc model for tis')
        svcs = self._get_dcmdl(self.svc_mdl, size, em_phases=2, max_splits=1)

        if arrs is None or svcs is None:
            return None
        else:
            v_arr, arr_avg, arr_std = arrs
            v_svc, svc_avg, svc_std = svcs
            if self.tta_mdl is None:
                tta_avg, tta_std = np.inf, np.nan
                models = [{'prob': p[0][0] * p[1][0], 'lbda': p[0][1], 'mu': p[1][1], 'theta': 0.0} for p in product(v_arr, v_svc)]
            else:
                s_ut.my_print('pid: ' + str(os.getpid()) + ' *********** dc model for tta')
                ttas = self._get_dcmdl(self.tta_mdl, size, em_phases=2, max_splits=1)
                if ttas is not None:
                    v_tta, tta_avg, tta_std = ttas
                    models = [{'prob': p[0][0] * p[1][0] * p[2][0], 'lbda': p[0][1], 'mu': p[1][1], 'theta': p[2][1]} for p in product(v_arr, v_svc, v_tta)]
                else:
                    return None

            m_models = [(m['prob'], self._get_erl_m(m['lbda'], m['mu'], m['theta'], prob=m['prob'])) for m in models]
            nulls = sum([m[0] for m in m_models if m[1] is None])
            m_opt = sum([m[0] * m[1] / (1.0 - nulls) for m in m_models if m[1] is not None])

            s_ut.my_print('pid: ' + str(os.getpid()) + ' dc result::lbda: ' + str(self.lbda) +
                  ' mu: ' + str(self.mu) + ' theta: ' + str(self.theta) + ' m: ' + str(int(m_opt)))

            # adjust when the dc result is much higher than the basic results
            # the value of max_delta is arbitrary
            if self.m_mdl is not None:
                max_delta = 1.5  # must be > 1 and less than 2
                if m_opt is not None:
                    if m_opt / self.m_mdl > max_delta:
                        m_opt = int(np.ceil((1 + 0.5 * (max_delta - 1)) * self.m_mdl))
                    elif m_opt / self.m_mdl < (1.0 / max_delta):
                        m_opt = int(self.m_mdl)
                    else:
                        m_opt = int(np.ceil(m_opt))
                else:
                    m_opt = int(np.ceil((1 + 0.5 * (max_delta - 1)) * self.m_mdl))
            else:
                m_opt = None if m_opt is None else int(np.ceil(m_opt))

            s_ut.my_print('pid: ' + str(os.getpid()) + ' final result::lbda: ' + str(self.lbda) +
                  ' mu: ' + str(self.mu) + ' theta: ' + str(self.theta) + ' m: ' + str(m_opt))
            util = max(1.0, svc_avg / (arr_avg * m_opt)) if m_opt is not None else None
            d_out = {'servers': m_opt, 'utilization': util,
                     'avg_tba': arr_avg, 'avg_tis': svc_avg, 'avg_tta': tta_avg,
                     'std_tba': arr_std, 'std_tis': svc_std, 'std_tta': tta_std
                     }
            return d_out


def solver(d_cfg):
    solver_obj = SimSolver(d_cfg)
    if solver_obj.is_valid:
        return solver_obj.get_dcmdl()
    else:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR: invalid models. Cannot solve')
        return None


if __name__ == '__main__':
    cfg_file = os.path.join(FILE_PATH, '../config/solver_cfg.json')
    # if len(sys.argv) > 1:
    #     cfg_file = os.path.expanduser(sys.argv[1])
    # else:
    #     s_ut.my_print(str(os.getpid()) + ' invalid arguments: ' + str(sys.argv))
    #     s_ut.my_print(str(os.getpid()) + ' ERROR')
    #     raise RuntimeError('failure')
    with open(cfg_file, 'r') as fp:
        d_cfg = json.load(fp)
    res = solver(d_cfg)
    s_ut.my_print(str(os.getpid()) + ' SUCCESS')


