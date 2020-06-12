"""
$ python examples/erlangH_sim.py
To change parameters go in the main section (in caps)
Erlang-H: simulation of Erlang-A with hyper-exponential service time and abandonments
This model can be used when the service time and abandons have heavy tails by modeling the heavy tailed distributions as a mixture of exponentials
(see ???)
- Erlang-H
  - set SVC_PROBS and SVC_TIMES as equal len arrays
  - set ABN_PROBS and ABN_TIMES as equal len arrays

- Regular Erlang-A: set SVC_PROBS = [1.0] and SVC_TIMES = [1/mu], ABN_PROBS=[1.0] and ABN_TIMES = [1/theta]

- Regular Erlang-C: set SVC_PROBS = [1.0] and SVC_TIMES = [1/mu] and ABN_PROBS=[1.0] and ABN_TIME = [np.inf]

"""

import os
import random
import sys
import simpy
import numpy as np
import pandas as pd
from collections import defaultdict
import time

import utilities.stats_utils as sut
import simulation.erlang_sim as sim_eh


if __name__ == '__main__':
    # parameters
    NUM_CUSTOMERS = 10000           # max number of customers to simulate
    VERBOSE = False
    idx_arr = None                  # boostrap array
    reuse = True                    # if true does not regenerate the random numbers over repeat runs
    RANDOM_SEED = 2

    # System parameters
    ARR_INTER = 1                 # inter-arrival-time, 1 / lambda
    NUM_SERVERS = 3               # m
    SVC_TIMES = [1, 10, 100]      # avg svc time for each component, 1 / mu_1, .., 1 / mu_K
    SVC_PROBS = [0.5, 0.4, 0.1]   # prob of selecting a svc rate, alpha_1, ..., alpha_K with alpha_1 + ... + alpha_K = 1
    ABN_TIMES = [5]               # avg time to abn for each component, i.e. 1 / theta_1, ..., 1 / theta_L. Set to [np.inf] for Erlang-C
    ABN_PROBS = [1]               # prob of selecting an abn  rate, beta_1, .., beta_L with beta_1 + ... + beta_L = 1

    # Performance parameters
    SLA_THRES = 0                 # use to compute the SLA quantile, q_sla = P(svc_time <= SLA_THRES & ~ABN)

    if len(SVC_TIMES) != len(SVC_PROBS) or sum(SVC_PROBS) != 1 or len(ABN_TIMES) != len(ABN_PROBS) or sum(ABN_PROBS) != 1:
        print('Invalid parameters')
        raise RuntimeError('failure')
    if np.prod(np.array(SVC_PROBS)) == 0 or np.prod(np.array(ABN_PROBS)) == 0:
        print('Cannot have 0 probs')
        raise RuntimeError('failure')

    # config
    avg_svc = np.mean(np.array(SVC_PROBS) * np.array(SVC_TIMES))
    avg_abn = np.mean(np.array(ABN_PROBS) * np.array(ABN_TIMES))
    print('\n**** SYSTEM PARAMETERS')
    print('inter-arrival time: ' + str(np.round(ARR_INTER, 2)))
    print('avg svc time: ' + str(np.round(avg_svc, 2)))
    print('avg abn time: ' + str(np.round(avg_abn, 2)))
    print('servers: ' + str(NUM_SERVERS))
    print('svc time classes: ' + str(len(SVC_TIMES)))
    print('svc time arrays: probs: ' + str(SVC_PROBS) + ' avgs: ' + str(SVC_TIMES))
    print('abn time classes: ' + str(len(ABN_TIMES)))
    print('abn time arrays: probs: ' + str(ABN_PROBS) + ' avgs: ' + str(ABN_TIMES))

    # Setup and start the simulation
    random.seed(RANDOM_SEED)
    env = simpy.Environment()

    # Start processes and run
    start = time.time()
    env.process(sim_eh.setup_n_run(env, NUM_SERVERS, ARR_INTER, SVC_PROBS, SVC_TIMES, ABN_PROBS, ABN_TIMES, NUM_CUSTOMERS, reuse=reuse, verbose=VERBOSE))
    env.run(until=None)
    print('DONE SIM')
    print('sim duration: ' + str(time.time() - start))

    # build results and compute estimates with boostrap
    start = time.time()
    results = defaultdict(list)
    for i in range(len(sim_eh.Customer.customer_list)):
        c = sim_eh.Customer.customer_list[i]
        results['is_abn'].append(c.is_abn)
        results['inter_arrival'].append(c.inter_arrival)
        results['svc_time'].append(c.svc_time)
        results['abn_time'].append(c.abn_time)
        results['q_sla'].append(1 if c.wait <= SLA_THRES else 0)
        results['occupancy'].append(sim_eh.Customer.arr_insvc[i] / NUM_SERVERS)
    data = pd.DataFrame(results)
    print('data prep duration: ' + str(time.time() - start))

    # bootstrap
    start = time.time()
    sple_fraction = 0.5
    n_iters = 1000
    n_samples = int(len(data) * sple_fraction)
    if idx_arr is None:
        idx_arr = [np.random.choice(range(len(data)), size=n_samples, replace=True) for _ in range(n_iters)]
    bs_df = sut.bootstrap(data, sple_fraction=sple_fraction, n_iters=n_iters, ci=0.95, idx_arr=idx_arr)
    print('boostrap duration: ' + str(time.time() - start))
    print(bs_df)


