"""
Server sizing: general class to find the number of servers needed for Erlang models
"""

import os
import numpy as np
from functools import lru_cache

from capacity_planning.queueing.erlang import erlang_queues as e_queues
from capacity_planning.utilities import sys_utils as s_ut


class ServerSizing(object):
    def __init__(self, mdl, mdl_pars, cons=None, window=4, verbose=False):
        # mdl = queueing model class name (ErlangA, ErlangB, ....)
        # mdl_pars = {'lbda': lbda, 'mu': mu, 'theta': theta, 'q': q, 'K': K}
        # constraint functions:
        #     ErlangA, ErlangA-PS: sla_prob(t) = prob(W<= t & !ABN), prob(ABN) <= abn, util <= u_max
        #     ErlangC: prob(W<= t), util <= u_max
        #     ErlangB: p(not-blocking), util <= u_max
        # target constraint functions
        #    utilization: u_max - utilization() >= 0
        #    B: 1 - p(blocking) >= q
        #    C: p(W<=t)  >= q
        #    A, A_PS: p(W<=t & !ABN) >= q & p(ABN) <= abn
        #    sla args = [q, t, abn] or [q, t] or [q]
        # m_bounds = (lwr_bnd, upr_bnd) bounds for m
        # cons: {          # dict of constraints for server sizing with format: constraint_name: {<func_name>: arg_list}
        #      'sla': {'sla_func': [sla_prob, sla_time]},          # func name, sla_args. We want prob(Wait > time) <= prob
        #      'abn': {'abn_func": [abn_prob]},                    # we want Prob(abn) <= abn_prob
        #      'util': {'util_func': [max_util]},                  # we want server_utilization <= max_util
        #       ...
        # }
        #
        self.mdl = mdl                                                # mdl class (ErlangA, ErlangB, ErlangC, ErlangA_PS)
        self.lbda = mdl_pars.get('lbda', None)                        # arrival rate
        self.mu = mdl_pars.get('mu', None)                            # svc rate
        self.theta = mdl_pars.get('theta', None)                      # abandon rate
        self.q = mdl_pars.get('q', None)                              # context switching
        self.K = mdl_pars.get('K', None)                              # max concurrency
        self.a = self.lbda / self.mu                                  # offered traffic
        self.window = window                                          # normalized upr bound for servers
        self.func_args, self.m_bounds = dict(), None
        self.verbose = verbose
        self.mval = None
        self.min_mval = 1         # gets reset for erlangC
        self.mdl_funcs = list()

        # constraints
        if cons is not None:
            self.server_cons = list()
            for d in cons.values():
                func_name = list(d.keys())[0]
                func_args = list(d.values())[0]
                if func_name == 'abn_func' and self.mdl != 'ErlangA':
                    continue
                if func_name == 'abn_func' and func_args == 1.0:   # not active: prob(ABN) <= p_ABN
                    continue
                if func_name == 'sla_func' and func_args[0] == 0.0:   # not active: prob(W <= t) >= q_SLA
                    continue
                if func_name == 'util_func' and func_args[0] == 1.0:   # not active: util <= max_utl
                    continue
                self.func_args[func_name] = func_args
                self.mdl_funcs.append(self.get_func(func_name))
                self.server_cons.append({'type': 'ineq', 'fun': lambda x: self.mdl_funcs[-1](self, x)})
        else:
            self.server_cons = None
        self.set_bounds()                                             # bounds for servers in the minimization

    def get_func(self, func_name):
        if func_name == 'sla_func':
            return self.sla_func
        elif func_name == 'abn_func':
            return self.abn_func
        elif func_name == 'util_func':
            return self.util_func
        else:
            return None

    def has_m(self):
        m = self.get_servers() if self.mval is None else self.mval
        if self.verbose is True:
            s_ut.my_print('WARNING: no value set for servers. Setting to default: ' + str(m))
        return m

    @lru_cache(maxsize=None)
    def utilization(self, m=None):
        if m is None:
            m = self.has_m()
        return min(1.0, self.queueing_mdl(m).utilization())

    @lru_cache(maxsize=None)
    def poor_svc_prob(self, m=None):  # P(W > t & !ABN)
        if m is None:
            m = self.has_m()
        t = self.func_args['sla_func'][-1]
        return self.queueing_mdl(m).poor_svc_prob(t)

    @lru_cache(maxsize=None)
    def abn_prob(self, m=None):  # P(ABN)
        if m is None:
            m = self.has_m()
        return self.queueing_mdl(m).abn_prob()

    @lru_cache(maxsize=None)
    def sla_prob(self, m=None):  # P(W <= t & !ABN)
        if m is None:
            m = self.has_m()
        t = self.func_args['sla_func'][-1]
        return self.queueing_mdl(m).sla_prob(t)

    def queue_class_switch(self):
        args = [self.lbda, self.mu]
        if self.mdl == 'ErlangA':
            if self.theta == 0.0:
                s_ut.my_print('WARNING: infinite patience: setting model to ErlangC')
                self.mdl = 'ErlangC'
                self.set_bounds()
                return e_queues.ErlangC, args
            else:
                args.append(self.theta)
                return e_queues.ErlangA, args
        elif self.mdl == 'ErlangB':
            return e_queues.ErlangB, args
        elif self.mdl == 'ErlangC':
            return e_queues.ErlangC, args
        elif self.mdl == 'ErlangA_PS':
            args.append(self.theta)
            args.append(self.q)
            args.append(self.K)
            return e_queues.ErlangA_PS, args
        else:
            s_ut.my_print('@@@@@@@ invalid model @@@@@@@: ' + str(self.mdl))
            return None

    @lru_cache(maxsize=None)
    def queueing_mdl(self, m):
        if isinstance(m, int) or isinstance(m, float):  # needs to be a list
            m = [m]
        queue_cls, mdl_args = self.queue_class_switch()
        args = list(mdl_args[:2]) + list(m) + list(mdl_args[2:])
        return queue_cls(*args, verbose=self.verbose)

    def set_bounds(self):
        if self.mdl == 'ErlangA':
            m_bounds = (max(1, self.a / self.window), self.window * self.a)   # overwrite bounds
        elif self.mdl == 'ErlangC':
            try:
                self.min_mval = int(np.ceil(self.a))
                if self.min_mval == int(self.a):
                    self.min_mval += 1
                m_bounds = (self.min_mval, self.window * self.a)   # bounds for servers in the minimization
            except ValueError:
                s_ut.my_print('ERROR_:erlang_tools:ErlangC model invalid::' + self.__str__())
                m_bounds = None
        elif self.mdl == 'ErlangB':
            m_bounds = (max(1, self.a / self.window), self.window * self.a)       # bounds for servers in the minimization
        else:
            s_ut.my_print('@@@@@@@ invalid model @@@@@@@: ' + str(self.mdl))
            m_bounds = None
        try:
            self.m_bounds = (int(np.floor(m_bounds[0])), int(np.ceil(m_bounds[1]))) if m_bounds is not None else None
        except ValueError:
            self.m_bounds = None

    @lru_cache(maxsize=None)
    def sla_func(self, m=None, err=False):
        # actual_sla - target_sla: if positive, we are meeting sla
        # find smallest m such that actual_sla >= target_sla
        # err: True return relative error wrt target prob
        if m is None:
            m = self.get_servers() if self.mval is None else self.mval
            if self.verbose is True:
                s_ut.my_print('WARNING: no value set for servers. Setting to default:: ' + str(m))
        q, t = self.func_args[self.sla_func.__name__]
        if m < self.min_mval:
            s_ut.my_print('WARNING: ' + self.mdl + ' sla_func could be unstable because m is too small: ' + str(m) + ' and min m: ' + str(self.min_mval))
            y = -q  # prob(Wait < t) = 0
        else:
            y = self.queueing_mdl(m).sla_prob(t) - q
            # s_ut.my_print('m: ' + str(m) + ' q: ' + str(q) + ' t: ' + str(t) +  ' prob: ' + str(self.queueing_mdl(m).sla_prob(t)) + ' ret: ' + str(y))
        # print('sla: ' + str(m) + ' ' + str(q) + ' ' + str(self.queueing_mdl(m).sla_prob(t, use_log=False)))
        if err is False:
            return y
        else:
            return y / q

    @lru_cache(maxsize=None)
    def abn_func(self, m=None, err=False):
        # abn_max - p(ABN). If positive, meeting SLA
        # err: True return relative error wrt target prob
        if m is None:
            m = self.get_servers() if self.mval is None else self.mval
            if self.verbose is True:
                s_ut.my_print('WARNING: no value set for servers. Setting to default:: ' + str(m))
        abn_max = self.func_args.get(self.abn_func.__name__, [1.0])[0]
        if m < self.min_mval:   # system not stable: abn_prob = 1
            s_ut.my_print('WARNING: ' + self.mdl + ' abn_func could be unstable because m is too small: ' + str(m) + ' and min m: ' + str(self.min_mval))
            y = abn_max - 1.0
        else:
            # print('abn: ' + str(m) + ' ' + str(abn_max) + ' ' + str(self.queueing_mdl(m).abn_prob(use_log=False)))
            y = abn_max - self.queueing_mdl(m).abn_prob(use_log=False)
        if err is False:
            return y
        else:
            return y / abn_max

    @lru_cache(maxsize=None)
    def util_func(self, m):
        # util gap between max_util and actual util. Positive meets sla
        max_util = self.func_args.get(self.util_func.__name__, [1.0])[0]
        if m < self.min_mval:  # util = 1.0
            s_ut.my_print('WARNING: ' + self.mdl + ' could be unstable because m is too small: ' + str(m) + ' and min m: ' + str(self.min_mval))
            y = max_util - 1.0
        else:
            y = max_util - self.queueing_mdl(m).utilization()
        return y

    def _get_servers(self, func):
        if self.m_bounds is None:
            return None
        else:
            mval = self.bisect(self.m_bounds[0], self.m_bounds[1], func)
            if mval is not None:
                return mval            # by definition it meets the constraint for func
            else:
                return None

    @lru_cache(maxsize=None)
    def bnds_adjust(self, m=None):
        if m is None:
            m = self.mval
        if m is None:
            m = self.min_mval
        if self.m_bounds is None:
            self.m_bounds = (m, 2 * m)
        else:
            if m < self.m_bounds[1]:
                self.m_bounds = (m, self.m_bounds[1])
            else:
                if self.window >= 1.0:
                    self.m_bounds = (m, max(self.window * m, m + 1)) if self.mval is None else (m, max(self.window * self.mval, m + 1))
                else:
                    self.m_bounds = (m, self.window * m) if self.mval is None else (m, self.window * self.mval)

    def get_servers(self, ctr=0):
        # all constraints will be met by m large enough
        res_arr = [self._get_servers(func) for func in self.mdl_funcs]
        if None in res_arr:  # could not find a solution for some of the constraints
            ctr += 1
            if ctr < 3:
                if self.m_bounds is None:
                    self.m_bounds = (self.min_mval, 2 * self.min_mval)
                else:
                    self.m_bounds = (max(self.min_mval, int(np.floor(self.m_bounds[0] / 2))), 2 * self.m_bounds[1])
                if self.verbose:
                    s_ut.my_print('could not get servers for all constraints. Expanding bounds: ' + str(self.m_bounds) + ' ctr: ' + str(ctr))
                return self.get_servers(ctr=ctr)
            else:
                if self.verbose:
                    s_ut.my_print('ERROR_: could not get servers after ' + str(ctr) + ' tries')
                    s_ut.my_print(self.__str__())
                    s_ut.my_print(res_arr)
                return None
        else:
            if len(res_arr) > 0:
                self.mval = max(res_arr)
                self.queueing_mdl(self.mval)
                return self.mval
            else:
                if self.verbose:
                    s_ut.my_print('degenerate case: no constraints')
                if self.mdl == 'ErlangC':
                    return self.min_mval
                else:
                    return 1

    def server_to_agent(self, shrinkage, occupancy, week_hrs, hoops):
        # hoops = (op_hours, op_days), ie. operational hours per day and operational days per week
        # week_hrs = (w_shifts, shift_hrs), i.e. weekly shifts and hours per shift is the number of hours an agent will work per week
        # basic equation: servers * util * op_hours * op_days = agents * week_hrs * occupancy * (1 - shrinkage)
        m = self.get_servers()
        util = self.utilization(m)
        if np.isnan(util):
            util = 1.0
        work_pct = (1 - shrinkage) * occupancy
        op_hrs, op_days = hoops
        w_shifts, shift_hrs = week_hrs
        shift_agents = np.ceil(m * util * op_hrs * op_days / (w_shifts * shift_hrs * work_pct))
        return m, shift_agents, util   # queueing servers, agents, server util

    def __str__(self):
        string = ''
        string += ' model: ' + str(self.mdl)
        string += ' lambda: ' + str(self.lbda)
        string += ' mu: ' + str(self.mu)
        if self.theta is not None: string += ' theta: ' + str(self.theta)
        if self.q is not None: string += ' q: ' + str(self.q)
        if self.K is not None: string += ' K: ' + str(self.K)
        string += ' a: ' + str(self.a)
        string += ' window: ' + str(self.window)
        string += ' m_bounds: ' + str(self.m_bounds)
        if self.server_cons is not None: string += ' constraints: ' + str(self.func_args)
        return string

    def bisect(self, x_min, x_max, func):
        # ################################################################
        # INPUT:
        # - Function f,
        # - endpoint values a, b
        # - tolerance TOL,
        # - maximum iterations NMAX
        # INPUT CONDITIONS:
        # - a < b, either f(a) < 0 and f(b) > 0 or f(a) > 0 and f(b) < 0
        # OUTPUT: value which differs from a root of f(x) = 0 by less than TOL
        #
        # N = 1
        # While  N ≤ NMAX  # limit iterations to prevent infinite loop
        #   c = (a + b) / 2  # new midpoint
        #   If  f(c) = 0 or (b – a) / 2 < TOL then  # solution found
        #       return (c)
        #       break
        #   N = N + 1  # increment step counter
        #   If sign(f(c)) = sign(f(a)) then
        #      a = c
        #   else
        #       b = c  # new interval
        # return ("Method failed.")  # max number of steps exceeded
        # ################################################################

        # assume x_min does not meet SLA and x_max meets SLA
        # func > 0 means SLA is met
        if func(x_min) > func(x_max):
            s_ut.my_print('ERROR_: bisect: ' + str(func.__name__) + ' is not monotonic for  m between xmin ' + str(x_min) + ' and xmax: ' + str(x_max))
            s_ut.my_print('queue model: ' + self.__str__())
            return None

        if x_max == x_min:
            x_max += 1
            x_min = max(x_min - 1, self.min_mval)

        ctr_max = max(1, np.ceil(np.log2(x_max - x_min)))
        ctr, x_last = 0, None
        while ctr < ctr_max:
            x = int(np.round((x_max + x_min) / 2, 0))
            if self.mdl == 'ErlangC':
                x = max(self.min_mval, x)
            if self.verbose:
                s_ut.my_print('ctr: ' + str(ctr) + ' xmax: ' + str(x_max) + ' xmin: ' + str(x_min) + ' new x: ' + str(x) + ' func: ' + str(func(x)))
            try:
                if func(x) >= 0:               # x meets SLA so solution is between x_min and x
                    if x - x_min <= 1:
                        ret = max(x, self.min_mval)
                        return ret
                    else:
                        x_max = x
                else:                          # x does not meet SLA: SLA is between x and x_max
                    if x_max - x <= 1:
                        ret = max(x_max, self.min_mval)
                        return ret
                    else:
                        x_min = x
            except OverflowError:            # ErlangC with small nbr of servers may be unstable
                s_ut.my_print(str(os.getpid()) + ' WARNING: bisect: unstable system for ' + str(x) + ' servers')
                x_min = x
            ctr += 1

        s_ut.my_print(str(os.getpid()) + 'ERROR_:bisect: too many iterations. Something went wrong. xmin: ' + str(x_min) + ' xmax: ' + str(x_max))
        # s_ut.my_print('pid: ' + str(os.getpid()) +
        #       ' start bisect: xmin: ' + str(x_min) + ' xmax: ' + str(x_max) + ' func: ' + str(func.__name__) +
        #       ' return: ' + str(None))
        return None


