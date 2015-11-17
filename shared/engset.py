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

import utilities.args as au

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

