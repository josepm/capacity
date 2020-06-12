"""
Engset Model: FCFS M/M/m/K/N
N jobs
m servers
K = number queueing + in service.
N - K = number of jobs "thinking"
NOTE: WE ASSUME that K >= N >= m and NO LOSS or RETRIES
System state: k = number in the queue (waiting + in svc)
Model parameters
  lambda(k) = lambda * (N - k)
  mu(k) = k * mu if 0 <= k <= m
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

Normalization
Sum_{k=0}^m p(k) = (1 + r)^N - r^(1 + m) Binomial[N, 1 + m] Hypergeometric2F1[1, 1 + m - N, 2 + m, -r]
Sum_{k=m+1}^N p(k) = e^(m/r) * (1 + m) * r^m * Binomial[N, 1 + m] * ExpIntegralE[1 + m - N, m/r]
ExpIntegralE[n, z] = integral_1^Infinity exp(-zt) / t^n dt
Hypergeometric2F1[a, b, c, z] = 2F1(a,b,c,z) = Sum_{k>=0} (a)_k (b)_k / (c)_k (z^k/k!)

"""

import numpy as np
import scipy.stats as stats


class MachineRepair:

    def __init__(self, lbda, mu, m, jobs):
        """
        avg_queue = avg # jobs in the queue
        avg_waiting = avg # jobs waiting for svc
        avg_in_svc = avg # jobs being served = avg # busy servers = service station utilization
        avg_thinking = avg # jobs in thinking state
        avg_queue = avg_waiting + avg_in_svc
        :param lbda: 1/lbda = think time
        :param mu:  1/mu = svc_time
        :param m: number of servers in the service station
        :param jobs: number of jobs in the system
        :return:
        """
        self.mu = mu
        self.lbda = lbda
        self.m = m
        self.r = lbda / np.float(mu)
        self.jobs = jobs
        if self.jobs == 0:
            print('no jobs')

        self.probs = np.zeros(1 + jobs, dtype=np.float)
        self.set_probs()

        # MR kpis
        self.avg_waiting = np.average(np.array([max(0, k - self.m) for k in range(0, 1 + self.jobs)]), weights=self.probs)  # number in repair station waiting for repair
        self.avg_in_svc = np.average(np.array([min(k, self.m) for k in range(0, 1 + self.jobs)]), weights=self.probs)       # avg nbr in svc or avg nbr busy servers
        self.avg_queue = self.avg_in_svc + self.avg_waiting
        self.avg_thinking = np.float(self.jobs) - self.avg_queue
        self.tput = self.lbda * self.avg_thinking                    # throughput (jobs/sec)
        self.cycle_time = np.float(self.jobs) / self.tput            # time for a job to complete a cycle (think time + queueing time + svc time)
        self.util = self.avg_in_svc / self.m                         # server utilization
        self.prob0 = self.probs[0]                                   # empty repair station
        self.prob_waiting = self.avg_waiting / np.float(self.jobs)
        self.prob_in_svc = self.avg_in_svc / np.float(self.jobs)
        self.prob_thinking = self.avg_thinking / np.float(self.jobs)

    def set_probs(self):
        self.probs[0] = 1.0
        _ = [self.get_prob(k) for k in range(1, 1 + self.jobs)]
        self.probs /= np.sum(self.probs)     # normalize

    def get_prob(self, k):
        self.probs[k] = self.probs[k-1] * self.r * (np.float(self.jobs) + 1.0 - k) / np.minimum(k, self.m)

    # def kpis(self, k):
    #     pk = self.probs[k]
    #     w = max(0, k - self.m)                # number waiting
    #     self.avg_waiting += w * pk            # avg jobs waiting at repair station
    #     bz = min(k, self.m)     # number of busy servers
    #     self.avg_in_svc += bz * pk            # avg jobs being processed by servers

    def wait_prob(self, t):   # prob(Wait > t)
        """
        p(Wait > t) = sum_{k>=m} prob(m) * Prob(Wait > t| wait_queue = k - m)  (the latter is a sum of 1 + k - m exp, ie Erlang)
        We ignore PASTA effects and so on
        :param t: wait time
        :return:
        """
        l_prob = [np.log(self.probs[k]) for k in range(0, self.m)] + \
                 [np.log(self.probs[k]) + stats.gamma.logsf(t, 1 + k - self.m, loc=0.0, scale=1.0 / (self.m * self.mu)) for k in range(self.m, 1 + self.jobs)]
        return sum(np.exp(l_prob))

    def __str__(self):
        s = 'lambda : ' + str(self.lbda) + ' mu: ' + str(self.mu) + ' servers (m): ' + str(self.m) + ' jobs: ' + str(self.jobs)
        s += ' cycle time: ' + str(self.cycle_time) + ' tput: ' + str(self.tput) + ' util: ' + str(self.util) + ' thinking: ' + str(self.avg_thinking / self.jobs) + ' pwr: ' + str(self.tput / self.cycle_time)
        return s


# example
if __name__ == "__main__":
    mr = MachineRepair(0.5, 1, 2, 10)
    print(mr)

    for r in [1, 10, 100]:
        for jobs in range(1, 100, 10):
            mr = MachineRepair(r, 1.0, 1, jobs)
            print(mr)


