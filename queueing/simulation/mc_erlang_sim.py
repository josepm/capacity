"""
MULTICLASS: WIP NOT DONE!!!
Erlang-sim: simulation of Erlang-A with arbitrary svc and abn distros
"""

# TODO: input multiclass support with single server class
# TODO: server multiclass support with routing rules for inputs
# TODO: general interarrival times

import os
import sys
import simpy
import numpy as np

class Servers(object):
    def __init__(self, env, n_servers, max_util=None):
        self.env = env
        self.a_server = simpy.Resource(env, n_servers)
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

    @classmethod
    def reset(cls, ru):
        cls.reuse = ru
        cls.customer_list = list()
        cls.arr_queue = list()  # queue seen just before arrivals
        cls.arr_insvc = list()  # queue seen just before arrivals

    @classmethod
    def init_cls(cls, n_customers, ru, t_inter, svc_mdl, tta_mdl, has_abns):
        cls.reuse = ru
        cls.n_customers = n_customers
        cls.has_abns = has_abns
        if cls.reuse == False or (cls.reuse == True and len(Customer.svc_times) == 0):
            cls.arr_times = np.random.exponential(t_inter, size=n_customers)
            cls.svc_times = svc_mdl.rvs(size=n_customers)
            if has_abns is True:
                cls.abn_times = tta_mdl.rvs(size=n_customers)

    def __init__(self, sim_env, cid, verbose):
        self.cid = cid
        self.verbose = verbose
        self.env = sim_env

        if self.reuse is None:
            print('class not initialized')
            raise RuntimeError('failure')

        self.inter_arrival = Customer.arr_times[self.cid]
        self.svc_time = Customer.svc_times[self.cid]
        self.abn_time = Customer.abn_times[self.cid] if Customer.has_abns is True else None

        self.is_abn = None
        self.wait = None
        self.arrive = self.env.now
        self.customer_list.append(self)

    def __str__(self):
        string = ' cid: ' + str(self.cid) + \
                 ' svc time: ' + str(np.round(self.svc_time, 4)) + \
                 ' abn time: ' + str(np.round(self.abn_time, 4))
        if self.wait == 0 or self.wait is None:
            return string
        else:
            return string + ' wait: ' + str(np.round(self.wait, 4))

    def print_msg(self, msg, end_msg=''):
        if self.verbose == True:
            print('time: ' + str(np.round(self.env.now, 4)) + msg + self.__str__() + end_msg)

    def request_server(self, servers):
        with servers.a_server.request() as server_request:

            # collect stats at arrival
            Customer.arr_queue.append(len(servers.a_server.queue))   # customers waiting
            Customer.arr_insvc.append(len(servers.a_server.users))    # customers in service

            # arrival
            self.print_msg(':: arrival ::', end_msg=' queue: ' + str(Customer.arr_queue[-1]))

            result = yield server_request | self.env.timeout(self.abn_time)
            if server_request in result:
                self.wait = self.env.now - self.arrive          # wait for svc
                self.print_msg('::svc_start::')
                yield self.env.process(servers.service(self))   # svc time
                self.is_abn = 0
                self.abn_time = np.nan
                self.print_msg('::departure::')
            else:   # abandonment
                if ~np.isinf(self.abn_time):
                    self.is_abn = 1
                    self.wait = np.nan                     # the wait = abn_time
                    self.print_msg(':: abandon ::')


def setup_n_run(sim_env, n_servers, t_inter, svc_mdl, tta_mdl, max_cnt, reuse=False, verbose=True):
    servers = Servers(sim_env, n_servers)
    cid = 0

    # check Erlang C model
    has_abns = False if tta_mdl is None else True
    if has_abns is False:                                  # Erlang-C: stability check: only for Erlang-C. Systems with ABNs are always stable.
        avg_svc = svc_mdl.stats(moments='m')
        if (1 / t_inter) * avg_svc / n_servers > 1:
            print('unstable Erlang C')
            return

    Customer.reset(None)                      # reset values for multiple runs
    Customer.init_cls(max_cnt, reuse, t_inter, svc_mdl, tta_mdl, has_abns)

    # initialize
    # load servers
    for i in range(n_servers):
        customer = Customer(sim_env, cid, verbose)
        sim_env.process(customer.request_server(servers))
        cid += 1

    # load queue
    for i in range(n_servers):
        customer = Customer(sim_env, cid, verbose)
        sim_env.process(customer.request_server(servers))
        cid += 1

    # run the simulation
    while cid < max_cnt:
        customer = Customer(sim_env, cid, verbose)
        sim_env.process(customer.request_server(servers))
        yield sim_env.timeout(customer.inter_arrival)
        cid += 1
    return

