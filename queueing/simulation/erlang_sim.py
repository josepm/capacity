"""
Erlang-sim: simulation of Erlang-A with arbitrary svc and abn distros
"""

import os
import simpy
import numpy as np

from capacity_planning.utilities import sys_utils as s_ut


class Servers(object):
    def __init__(self, env, n_servers, max_util=None):
        self.env = env
        self.a_server = simpy.Resource(env, max(1, n_servers))
        self.time_worked = 0
        self.jobs_done = 0

    def service(self, c_obj):
        svc_time = c_obj.svc_time
        yield self.env.timeout(svc_time)
        self.time_worked += svc_time
        self.jobs_done += 1


class Customer(object):
    arr_times = list()
    svc_times = list()
    abn_times = list()
    has_abns = None
    reuse = None
    n_customers = None
    customer_list = list()
    arr_queue = list()    # queue seen just before arrivals
    arr_insvc = list()    # queue seen just before arrivals
    cls_init = False

    @classmethod
    def reset(cls, ru):
        cls.reuse = ru
        cls.customer_list = list()
        cls.arr_queue = list()  # queue seen just before arrivals
        cls.arr_insvc = list()  # queue seen just before arrivals

    @classmethod
    def init_cls(cls, n_customers, ru, arr_mdl, svc_mdl, tta_mdl, has_abns):
        cls.reuse = ru
        cls.n_customers = n_customers
        cls.has_abns = has_abns
        if cls.reuse == False or (cls.reuse == True and cls.cls_init == False):
            cls.cls_init = True
            cls.arr_times = arr_mdl.rvs(size=n_customers)
            cls.svc_times = svc_mdl.rvs(size=n_customers)
            cls.abn_times = tta_mdl.rvs(size=n_customers) if tta_mdl is not None else np.array([np.inf] * n_customers)

        if np.min(cls.svc_times) < 0 or np.min(cls.abn_times) < 0:
            s_ut.my_print(str(os.getpid()) + ' invalid values svc_time: ' + str(np.min(cls.svc_times)) + ' abn_time: ' + str(np.min(cls.abn_times)))
            s_ut.my_print(str(os.getpid()) + ' svc_time:: ' + svc_mdl.dist_name + ' params: ' + str(svc_mdl.params))
            s_ut.my_print(str(os.getpid()) + ' abn_time:: ' + tta_mdl.dist_name + ' params: ' + str(tta_mdl.params))

    def __init__(self, sim_env, cid, verbose):
        self.cid = cid
        self.verbose = verbose
        self.env = sim_env

        if self.reuse is None:
            s_ut.my_print(str(os.getpid()) + ' class not initialized')
            raise RuntimeError('failure')

        if self.cid < self.n_customers:
            try:
                self.inter_arrival = Customer.arr_times[self.cid]
                self.svc_time = Customer.svc_times[self.cid]
                self.abn_time = Customer.abn_times[self.cid]
            except IndexError:
                s_ut.my_print('ERROR_: ctr cid: ' + str(self.cid) + ' ttl customers: ' + str(self.n_customers) )
        else:
            s_ut.my_print(str(os.getpid()) + ' WARNING: counter mismatch: ' + str(self.cid) + ' ' + str(self.n_customers))

        self.is_abn = None
        self.wait = None
        self.arrive = self.env.now
        self.customer_list.append(self)

    def __str__(self):
        string = ' cid: ' + str(self.cid) + ' svc time: ' + str(np.round(self.svc_time, 4)) + ' abn time: ' + str(np.round(self.abn_time, 4))
        if self.wait == 0 or self.wait is None:
            return string
        else:
            return string + ' wait: ' + str(np.round(self.wait, 4))

    def print_msg(self, msg, end_msg='', detail=False):
        if self.verbose == True and detail == True:
            s_ut.my_print(str(os.getpid()) + ' time: ' + str(np.round(self.env.now, 4)) + msg + self.__str__() + end_msg)
            pass

    def request_server(self, servers, detail=False):
        with servers.a_server.request() as server_request:

            # collect stats at arrival
            Customer.arr_queue.append(len(servers.a_server.queue))   # customers waiting
            Customer.arr_insvc.append(len(servers.a_server.users))    # customers in service

            # arrival
            self.print_msg(' :: arrival ::', end_msg=' queue: ' + str(Customer.arr_queue[-1]), detail=detail)

            try:
                result = yield server_request | self.env.timeout(self.abn_time)
            except ValueError:
                s_ut.my_print(str(os.getpid()) + ' neg delay')
                s_ut.my_print(str(os.getpid()) + ' ' + str(server_request))
                s_ut.my_print(str(os.getpid()) + ' ' + str(self.abn_time))
                s_ut.my_print(str(os.getpid()) + ' ' + self.__str__())
                raise RuntimeError('failure')

            if server_request in result:
                self.wait = self.env.now - self.arrive          # wait for svc
                self.print_msg(' ::svc_start::', detail=detail)
                yield self.env.process(servers.service(self))   # svc time
                self.is_abn = 0
                self.abn_time = np.nan
                self.print_msg(' ::departure::', detail=detail)
            else:   # abandonment
                if ~np.isinf(self.abn_time):
                    self.is_abn = 1
                    self.wait = np.nan                     # the wait = abn_time
                    self.print_msg(' :: abandon ::', detail=detail)


def setup_n_run(sim_env, n_servers, q_cnt, arr_mdl, svc_mdl, tta_mdl, max_cnt, max_sim, reuse=False, verbose=True, detail=True):
    servers = Servers(sim_env, n_servers)
    cid = 0

    # check Erlang C model
    has_abns = False if tta_mdl is None else True
    if has_abns is False:                                  # Erlang-C: stability check: only for Erlang-C. Systems with ABNs are always stable.
        avg_svc = svc_mdl.avg   # svc_mdl.stats(moments='m')
        avg_arr = arr_mdl.avg   # arr_mdl.stats(moments='m')
        rho = (1 / avg_arr) * (avg_svc / n_servers)
        if rho > 1:
            s_ut.my_print(str(os.getpid()) + ' unstable Erlang C: scv: ' + str(avg_svc) + ' lbda: ' + str(1 / avg_arr) + ' servers: ' + str(n_servers) + ' rho: ' + str(rho))
            return

    # s_ut.my_print('pid: ' + str(os.getpid()) + ' ****** sim starts::servers: ' + str(n_servers) + ' init queue: ' + str(q_cnt) +
    #       ' customers: ' + str(max_cnt) +
    #       ' tba dist fam:  ' + arr_mdl.dist_name +
    #       ' tis dist fam: ' + svc_mdl.dist_name +
    #       ' tta dist fam: ' + str(None if tta_mdl is None else tta_mdl.dist_name))

    Customer.reset(None)                      # reset values for multiple runs
    Customer.init_cls(max_cnt + q_cnt, reuse, arr_mdl, svc_mdl, tta_mdl, has_abns)

    # initialize to avg queue
    # s_ut.my_print(str(os.getpid()) + ' ****** loading queue')
    for i in range(q_cnt):
        customer = Customer(sim_env, cid, verbose)
        sim_env.process(customer.request_server(servers, detail=detail))
        cid += 1

    # run the simulation
    # s_ut.my_print(str(os.getpid()) + ' ****** start sim')
    while cid < max_sim + q_cnt:
        customer = Customer(sim_env, cid, verbose)
        sim_env.process(customer.request_server(servers, detail=detail))
        yield sim_env.timeout(customer.inter_arrival)
        cid += 1
    return

