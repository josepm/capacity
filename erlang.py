
"""
################################
# Implements Erlang B formulas
################################
erlB(servers, traffic)
servers(traffic, prob)
traffic(servers, prob)
p(m, rho) = (rho^m/m!)/Sum[rho^i/i!, {i, 0, m}]
rho = a = XR
X: xtions/sec
R = resp time (in secs)
################################
Algorithm
  assume rho >= 0 and m >= 0 and m integer
  return 0 if rho == 0
  s = 0
  for i = 1 to m
    s = (1 + s) * (i/rho)
  end
return 1 / (1 + s)
"""


def erlB(m, rho):
    """
    # m = # servers, rho = avg # of active connections
    :param m: number of servers
    :param rho: avg number of concurrent requests (connections, threads, ...) rho = lambda / mu offered traffic
    :return: prob of blocking
    """
    if rho == 0:
        return 0
    if isinstance(m, int) and rho > 0:
        a = float(rho)
        r = 1.0
        for n in range(1, m + 1):
            r = 1 + r * n / a
        return 1.0 / r
    else:           # interpolate
        m_ceil = np.ceil(m).astype(int)
        m_floor = np.floor(m).astype(int)
        return erlB(m_floor, rho) + (m - m_floor) * (erlB(m_ceil, rho) - erlB(m_floor, rho))


def ext_erlB(m, rho, retry=0.0, eps=1E-6):
    """
    extended erlang B with retries
    r = rho
    a = r * erlB(m, r)
    r = rho + p_retry * a
    :param m: servers
    :param rho: concurrent requests
    :param p_retry: prob of retry of blocked
    :param eps: relative offered load error
    :return:
    """
    old_r, r = rho, rho
    delta = eps
    while delta >= eps:
        b = erlB(m, r)
        a = r * b
        r = rho + retry * a
        delta = (r - old_r) / old_r
        old_r = r
    return b * (1.0 - retry) / (1.0 - b * retry)  # prob loosing a call


def erlC(m, rho):
    """
    erlang C formula
    :param m: number of servers
    :param rho: lambda/mu offered traffic. Must be <= m
    :return: probability of queueing
    """
    if rho >= m:
        return 1.0
    elif m == 0:
        return 1.0
    else:
        b = erlB(m - 1, rho)
        d = (1.0 - rho) / (rho * b)
        return 1.0 / (1.0 + d)


def F_b(m, rho):
    """
    load carried by the last server (marginal load)
    :param m: servers
    :param rho: concurrent requests
    :return:
    """
    return rho * (erlB(m - 1, rho) - erlB(m, rho))


def servers(rho, pblock, m_init=16384):
    """
    compute the smallest # of servers m for given rho and blocking probability pblock such that erlB(m, rho) < pblock < erlB(m-1, rho)
    p(m, rho) decreases with m
    Given rho and a blocking prob B, find the largest m such that p(m) <= B
    bh < pblock and bl > pblock and h > l (the high number h provides the lower bound and vice-versa)
    :param rho: offered traffic
    :param pblock: blocking prob
    :return:  number of servers
    """

    if pblock > 1.0 or pblock < 0.0 or rho < 0.0:
        print 'Invalid parameters. prob: ' + str(pblock) + ' load: ' + str(rho)
        sys.exit(0)

    if pblock == 1.0:
        return 0

    if pblock == 0.0:
        return np.nan

#   bounds for m
    b = erlB(m_init, rho)
    m_min = m_init
    while b <= pblock:
        m_min = np.floor(m_min / 2.0)
        b = erlB(m_min, rho)
        if m_min == 1:
            break
    m_max = 2 * m_min   # upper bound

#   do it the dumb way (no bissection)
    for m in range(m_min, m_max):
        b = erlB(m, rho)
        if b <= pblock:
            b1 = b
            b2 = erlB(m - 1, rho)
            return m - (pblock - b1) / (b2 - b1)
    print 'Invalid range'    # in case we ever get here
    sys.exit(0)


# increases with servers and pblock
# computes a
def traffic(servers, pblock):
    """
    computes the max traffic given servers and pblock
    :param servers:
    :param pblock:
    :return:
    """
    if pblock <= 0 or servers <= 0 or pblock >= 1:
      print 'Invalid parameters. prob: ' + str(pblock) + ' servers: ' + str(servers)
      sys.exit(0)

#   find a starting point that is reasonable
    a = servers * np.exp(np.log(pblock) / float(servers))
    b = erlB(servers, a)
    u, l = 0, 0
    if b > pblock:
        u = a   # we found an upper bound
        while b > pblock:
          a /= 2.0
          b = erlB(servers, a)
        l = a
    else:      # we found a lower bound
        while b <= pblock:
          a *= 2.0
          b = erlB(servers, a)
        u = a

    iter, m = 0, 0
    while u - l > 1.0e-6 and iter < 500:
        iter += 1
        m = (u + l) / 2.0
        b = erlB(servers, m)
        if b > pblock:   # m is an upper bound
            u = m
        else:
            l = m
    return m


def erlBBulk(m, rho, bsz):
    """
    # m = # servers, rho = avg # of active connections
    :param m: number of servers
    :param rho: avg number of active requests (connections, threads, ...)
    :param bsz: avg bulk size
    :return: prob of blocking
    """
    if rho == 0:
        return 0
    if bsz <= 1:
      return erlB(m, rho)

    q = (bsz - 1.0) / np.float(bsz)
    probs = np.zeros(m + 1, dtype=float)
    psum = 0.0
    bprod = 1.0
    probs[0] = 1.0 / np.float(m)
    for k in range(1, m + 1):
        probs[k] = probs[k-1] * (rho * (1.0 - q) + q * (k - 1)) / np.float(k)
        psum += probs[k]
        bprod *= ((1.0 - q) * rho + q * k) / np.float(k)
    p0 = 1.0 - np.float(m) * psum
    return p0 * bprod


import numpy as np
import sys
import os
import json

# add the path to utilities
f = os.path.dirname(os.path.realpath(__file__))
par_dir = os.path.abspath(os.path.join(f, os.pardir))           # parent dir
gp_dir = os.path.abspath(os.path.join(par_dir, os.pardir))       # grand-parent dir
sys.path.append(par_dir)
sys.path.append(gp_dir)

import args as au

"""
Engset Model: FCFS M/M/m/K/N
N jobs
m servers
K = queueing + in service.
NOTE: WE ASSUME that K >= N >= m and NO LOSS or RETRIES
System state: k = number in the queue (waiting + in svc)
Model parameters
  lambda(k) = lambda * (N - k) if 0 <= k < K
  mu(k) = k * mu if 0<= k <= m
  mu(k) = m * mu if k > m
  r = lambda / mu
  p(k) = prob k in the multiserver queue (waiting + in svc)
B(n, k): binomial coefficient
B(n, k) = B(n-1, k-1) + B(n-1, k) with B(n, 0) = 1 and B(0, k) = 0
p(k) = p(0) * r^k * B(N, k)                            0 <= k <= m
p(k) = p(0) * r^k * B(N, k) * (k!/m!) * m^(m-k)        m <= k <= K
     = p(m) * (N-m)! / (N - k)! * r^(k - m) * m^(m-k)   m <= k <= K
Recursions
f(0, N) = 1, N >= 0
f(k, N) = f(k, N-1) * N / (N - k)          0 <= k < N, N >= 1
p(k, N) = f(k, N) / Sum_{k=0}^N f(k, N)
p(k, N) = p(k-1, N) * r * (N+1-k) / k                0 < k <= m
p(k, N) = p(k-1, N) * r * (N+1-k) / m                    m < k
p(k, N) = p(k-1, N) * r * (N+1-k) / min(k, m)         0 < k <= N
Not implemented recursions
f(0, N) = 1, N >= 0
f(k, N) = r * f(k -1, N) * (N - (k-1)) / min(m, k)        0 < k <= N, N >= 1
s(N) = sum_{k=0}^N f(k, N)
p(k, N) = f(k, N) / s(N)
avg queue
  avg(0) = 0
  avg(N) = N (1 - s(N - 1) / s(N))
queue variance
  var(0) = 0
  var(N) = (avg(N) - avg(N-1)) * (N - avg(N))
States of ONE job:
  - S = being served: Moves out of the state at a rate mu.
      P(S) = prob a job is in state S
      L(S) = avg # jobs in state S = avg # busy servers
        L(S) = Sum_{k=0}^m k * p(k) + m * Sum{k=m+1}^N p(k)
        P(S) = L(S) / N
  - T = thinking/not runnable: Moves out of the state at a rate lambda.
        P(T) = prob a job is in thinking state
        L(T) = avg # jobs in thinking state
        L(T) = Sum_{k=0}^N p(k) * (N-k)
        P(T) = L(T) / N
  - W = waiting in the queue: Moves out of the state at a rate which is NOT exponential (Erlang distribution)
        P(W) = prob a job has to wait for service
        L(W) = avg # jobs waiting
        L = Sum_{k=0}^N k * p(k)  avg # jobs in the queue
        L(W) = L - L(S)
        P(W) = L(W) / N
  - We also have: lambda * P(T) = mu * P(S)
"""


class MachineRepair:

    def __init__(self, lmbda, mu, servers, jobs):
        """
        avg_queue = avg # jobs in the queue
        avg_waiting = avg # jobs waiting for svc
        avg_insvc = avg # jobs being served = avg # busy servers
        avg_thinking = avg # jobs in thinking state
        avg_queue = avg_waiting + avg_insvc
        :param lmbda: 1/lmbda = think time
        :param mu:  1/mu = svc_time
        :param servers:
        :param jobs:
        :return:
        """
        self.mu = mu
        self.lmbda = lmbda
        self.servers = servers
        self.r = lmbda / np.float(mu)
        self.jobs = jobs
        if self.jobs == 0:
            print 'no jobs'
            sys.exit(0)
        self.probs = np.zeros(1 + jobs, dtype=np.float)
        self.probs[0] = 1.0
        _ = [self.get_prob(k) for k in range(1, 1 + self.jobs)]
        self.probs /= np.sum(self.probs)
        self.avg_queue = np.average(np.arange(0, 1 + self.jobs), weights=self.probs)

        # more MR kpis
        self.avg_waiting, self.avg_in_svc, self.var_queue = 0.0, 0.0, 0.0
        _ = [self.kpis(k) for k in range(1 + self.jobs)]
        self.avg_thinking = np.float(self.jobs) - self.avg_queue
        self.tput = self.lmbda * self.avg_thinking                  # throughput (jobs/sec)
        self.cycle_time = np.float(self.jobs) / self.tput                     # time for a job to complete a cycle (think time + queueing time + svc time)
        self.util = self.avg_in_svc / self.servers                   # server utilization
        self.prob0 = self.probs[0]                              # empty repair station
        self.prob_waiting = self.avg_waiting / np.float(self.jobs)
        self.prob_in_svc = self.avg_in_svc / np.float(self.jobs)
        self.prob_thinking = self.avg_thinking / np.float(self.jobs)

    def get_prob(self, k):
        self.probs[k] = self.probs[k-1] * self.r * (np.float(self.jobs) + 1.0 - k) / np.minimum(k, self.servers)

    def kpis(self, k):
        pk = self.probs[k]
        w = k - self.servers if k > self.servers else 0   # number waiting
        self.avg_waiting += w * pk                        # avg jobs waiting at repair station
        bz = k if k <= self.servers else self.servers     # number of busy servers
        self.avg_in_svc += bz * pk                         # avg jobs being processed by servers

if __name__ == "__main__":
    arg_dict = au.get_pars(sys.argv[1:])
    servers, think_time, processing_time = arg_dict['servers'], arg_dict['think_time'], arg_dict['processing_time']
    s_out = 'jobs, servers, r, queue, pwr, tput\n'
    for s in servers:
        for p in processing_time:
            for jobs in range(1, 4 * s + 1):
                    MR = MachineRepair(1.0 / think_time, 1.0 / p, s, jobs)
                    d = {'servers': s, 'r': MR.r}
                    tput = (jobs - MR.avg_queue) / np.float(think_time)
                    pwr = MR.r * (float(jobs) / MR.avg_queue - 1.0)
                    d['jobs'] = jobs
                    d['queue'] = MR.avg_queue
                    d['pwr'] = pwr
                    d['tput'] = tput
                    s_out += json.dumps(d) + '\n'
    with open('/tmp/engset.json', 'w') as f:
        f.write(s_out)


