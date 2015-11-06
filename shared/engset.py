import numpy as np
import sys

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
  p(k) = prob k in the multiserver queue

B(n, k): binomial coefficient
B(n, k) = B(n-1, k-1) + B(n-1, k) with B(n, 0) = 1 and B(0, k) = 0
p(k) = p(0) * r^k * B(N, k)                            0 <= k <= m
p(k) = p(0) * r^k * B(N, k) * (k!/m!) * m^(m-k)        m <= k <= K
     = p(m) * (N-m)! / (N- k)! * r^(k - m) * m^(m-k)   m <= k <= K

p(k, N) = f(k, N) / Sum_{k=0}^N f(k, N)

Recursions
f(0, N) = 1, N >= 0
f(k, N) = f(k, N-1) * N / (N - k)          0 <= k < N, N >= 1

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
        self.jobs = np.float(jobs)
        r = lmbda / np.float(mu)

        self.avg_queue, self.avg_waiting, self.avg_insvc, self.var_queue = 0.0, 0.0, 0.0, 0.0
        f = []
        f[0], sum = 1.0, 1.0  / self.jobs    # divide sum by # jobs to improve converge at large values
        for k in range(1, int(self.jobs) + 1):
            min_k = np.float(k) if k <= self.servers else self.servers
            f.append(f[- 1] * r * (self.jobs - (k - 1)) / min_k)
            sum += f[-1] / self.jobs                               # divide by # jobs to improve converge at large values
            if np.isnan(sum):
                print 'overflow'
                sys.exit(0)

        self.probs = []
        for k in range(len(f)):
            pk = (f[k] / sum) / self.jobs
            self.probs[k] = pk
            self.avg_queue += k * pk
            self.mr_kpis(k, pk)         # self.avg_waiting, self.avg_insvc

        for k in range(len(f)):        # queue variance
            self.var_queue += ((k - self.avg_queue) * (k - self.avg_queue)) * self.probs[k]

        # more MR kpis
        self.avg_thinking = self.jobs - self.avg_queue
        self.tput = self.lmbda * (self.jobs - self.avg_queue)          # throughput (jobs/sec)
        self.cycle_time = self.jobs / self.tput                     # time for a job to complete a cycle (think time + queueing time + svc time)
        self.util = self.avg_insvc / self.servers                   # server utilization
        self.prob0 = self.probs[0]                              # empty repair station
        self.prob_waiting = self.avg_waiting / self.jobs
        self.prob_insvc = self.avg_insvc / self.jobs
        self.prob_thinking = self.avg_thinking / self.jobs

    def mr_kpis(self, k, pk):
        w = k -self.servers if k > self.servers else 0   # number waiting
        self.avg_waiting += w * pk                       # avg jobs waiting at repair station
        bz = k if k <= self.servers else self.servers        # number of busy servers
        self.avg_insvc += bz * pk                        # avg jobs being processed by servers
