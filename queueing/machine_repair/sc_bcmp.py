"""
Usage:
$ python sc_bcmp.py config_file.json

Based on http://nicky.vanforeest.com/queueing/multiServerConvolutionAlgorithm/multiServerConvolutionAlgorithm.html
[ABGdMT06]	G. Bolch, S. Greiner, H. de Meer, and K. S. Trivedi. Queueing Networks and Markov Chains: Modeling and Performance Evaluation with Computer Science Applications. John Wiley & Sons, 2006.
Each function corresponds to a formula with the same name in as in Bolch et al. [ABGdMT06].
The numbers in the quotes refer to equations in [ABGdMT06].
[ABGdMT06] numbers the stations from 1 on; here we start at 0 (conforming to the Python, and C, standard).
This implies that below we sometimes count to numStations - 1 rather than numStations.

Changes:
- computations done in logs to allow better scalability
- recursions removed to allow better scalability. No trampolin, just an init phase (that may be slow) to compute the basic numbers in the class ranges. 
  Can study changes in performance with number of jobs with the same network instance.
- drop memoization as it is not needed

Test Cases:
Example 8.1, Figure 8.3 of [ABGdMT06].
e = [1, 2./3, 0.2]       # visit ratios
m = [2, 3, 1]            # number of servers at a station
mu = [0.8, 0.6, 0.4]     # service rates
jobs = 3
c = BCMP(name='Ex1', e=e, m=m, mu=mu,jobs=jobs)
c.cycleTime() = 3.1760864152362367
c.avg_jobs(2) = 0.6604480790868028
c.avg_response_time(2) = 3.4960669532607698

Penny Fab number 2 of [AHS08]. A comparison to Figure 7.12 of Factory Physics shows that the results are the same as in FP.
[AHS08]	W.J. Hopp and M.L. Spearman. Factory Physics. Waveland Press, Inc., 3rd edition, 2008.
e = [1, 1, 1, 1]
m = [1, 2, 6, 2]
mu = [1./2, 1./5, 1./10, 1./3]
jobs = 80
c = BCMP(name='Ex2', e=e, m=m, mu=mu, jobs=jobs)
c.cycleTime() = 200.00000514076604
c.avg_jobs(2) = 4.569521342432143
c.avg_response_time(2) = 11.423803649966926
"""

import numpy as np
import math
import json
import os
import sys
from functools import lru_cache

class BCMP(object):
    def __init__(self, name, e, m, mu, jobs, use_logs=True, iter=True):
        """
        the len(e) = len(m) = len(mu) = number of stations in the network: from 0 to stations - 1
        :param e: visit ratios: avg number of visits to each station by a job.
                  It is the solution of e = eR with r_ij = prob a job goes from station i to station j, setting e[0] = 1 to get a unique solution.
        :param m: servers in each station
        :param mu: service rate at each station
        :param jobs: jobs in the system
        :param use_logs: use log based functions if True
        :param iter:
        :param use_logs: use log based functions if True
        :return:
        """
        self.name = name
        self.e = [float(r) for r in e]
        self.m = [int(s) for s in m]
        self.mu = [float(r) for r in mu]
        self.numJobs = int(jobs)
        self.numStations = len(self.mu)
        self.use_logs = use_logs

        self.G_dict = dict()
        self.GN_dict = dict()
        self.probs_dict = dict()

        if self.numJobs <= 0:
            print('invalid jobs: ' + str(self.numJobs))

        # initialize
        # implement equation 8.4
        for station in range(self.numStations):
            self.G_dict[station] = dict()
            for jobs in range(self.numJobs + 1):
                self.G_dict[station][jobs] = self.G(station, jobs)

        if iter is True:
            # print self.name + ': G completed'

            # note before eq 8.8. Not clear about eq 8.11
            for idx in range(len(self.e) - 1):  # for last station use G[N-1][k] which is already done
                # drop station idx
                e_idx = self.e[:idx] + self.e[(idx + 1):]   # self.drop_station(self.e, idx)
                m_idx = self.m[:idx] + self.m[(idx + 1):]   # self.drop_station(self.m, idx)
                mu_idx = self.mu[:idx] + self.mu[(idx + 1):]   # self.drop_station(self.mu, idx)
                self.GN_dict[idx] = dict()
                c_idx = BCMP(name=str(idx), e=e_idx, m=m_idx, mu=mu_idx, jobs=self.numJobs, use_logs=True, iter=False)
                # print 'iteration for ' + self.name + ' stations (servers): ' + str(c_idx.m) + ' dropped station: ' + str(idx)
                self.GN_dict[idx] = {k: c_idx.G_dict[len(e_idx) - 1][k] for k in c_idx.G_dict[len(e_idx) - 1]}

            # last station (see comment between eq 8.8 and 8.9)
            self.GN_dict[len(self.e) - 1] = {k: self.G_dict[len(self.e) - 2][k] for k in self.G_dict[len(self.e) - 2]}
            # print 'iteration for ' + self.name + ' stations (servers): ' + str(self.m[:-1]) + ' dropped station: ' + str(len(self.m)-1)
            # print self.name + ': GN completed'

            # compute probs
            for station in range(self.numStations):
                self.probs_dict[station] = dict()
                for jobs in range(self.numJobs + 1):
                    self.probs_dict[station][jobs] = self.prob(station, jobs)
                s_probs = sum([p for p in self.probs_dict[station].values()])
                if np.abs(s_probs - 1.0) > 1e-08:
                    print('BCMP failed for network ' + str(self.name) + ' ' + str(self.numJobs) + ' ' + str(self.m) + ' ' + str(self.mu) + ' probs: ' + str(s_probs))
            # print self.name + ': probs completed'

    def __str__(self):
        string = 'jobs: ' + str(self.numJobs) + ' cycle_time: ' + str(self.cycleTime()) + '\n'
        for i in range(len(m)):
            string += 'station: ' + str(i) + \
                      ' jobs_waiting: ' + str(self.avg_jobs_waiting(1)) + \
                      ' busy_servers: ' + str(self.avg_busy_servers(i)) + \
                      ' tput: ' + str(self.tput(i)) + \
                      ' response_time: ' + str(self.avg_response_time(i))
        return string

    @lru_cache()
    def beta(self, i, k):
        if self.use_logs:
            return self._log_beta(i, k)
        else:
            return self._beta(i, k)

    @lru_cache()
    def _log_beta(self, i, k):   # Bolch 7.62 (log)
        """
        return log(beta(i, k))
        :param i: station
        :param k: number of jobs
        :return:
        """
        if self.m[i] == 1:
            return 0
        if k <= self.m[i]:
            return sum([np.log(x) for x in range(1, 1 + k)])
        if k > self.m[i]:
            return sum([np.log(x) for x in range(1, 1 + int(self.m[i]))]) + (k - self.m[i]) * np.log(self.m[i])

    @lru_cache()
    def _beta(self, i, k):  # Bolch 7.62
        """
        return beta(i, k)
        :param i: station
        :param k: number of jobs
        :return:
        """
        if self.m[i] == 1:
            return 1
        if k <= self.m[i]:
            return math.factorial(k)
        if k > self.m[i]:
            return math.factorial(self.m[i]) * (self.m[i]**(k-self.m[i]))

    @lru_cache()
    def F(self, i, k):
        return self.log_F(i, k) if self.use_logs else self._F(i, k)

    @lru_cache()
    def log_F(self, i, k):    # Bolch 7.61 (log)
        """
        :param i: station in network
        :param k: number of jobs in the station
        :return:
        """
        return k * np.log(self.e[i]) - k * np.log(self.mu[i]) - self.beta(i, k)

    @lru_cache()
    def _F(self, i, k):    # Bolch 7.61
        """
        :param i: station in network
        :param k: number of jobs in the station
        :return:
        """
        return (self.e[i] / self.mu[i])**k / self.beta(i, k)

    @lru_cache()
    def G(self, i, k):
        return self._log_G(i, k) if self.use_logs else self._G(i, k)

    @lru_cache()
    def _log_G(self, n, k):
        """
        returns log(G(n, k))
        :param n: station (0 to N-1)
        :param k: jobs in the network
        :return:
        """
        if k == 0:  # Bolch 8.6
            return 0.0
        if n == 0:  # Bolch 8.5
            return self.log_F(0, k)
        else:       # Bolch 8.4
            val_list = [self.log_F(n, j) + self.G_dict[n - 1][k - j] for j in range(k + 1)]
            max_val = max(val_list)
            d_list = [x - max_val for x in val_list]
            return max_val + np.log(sum([np.exp(x) for x in d_list]))

    @lru_cache()
    def _G(self, n, k):
        """
        :param n: station (0 to N-1)
        :param k: jobs in the network
        :return:
        """
        if k == 0:  # Bolch 8.6
            return 1.0
        if n == 0:  # Bolch 8.5
            return self._F(0, k)
        else:       # Bolch 8.4
            return sum([self._F(n, j) * self._G(n - 1, k - j) for j in range(k+1)])

    @lru_cache()
    def GN(self, i, k):
        return self._log_GN(i, k) if self.use_logs else self._GN(i, k)

    @lru_cache()
    def _log_GN(self, i, k):
        """
        returns log(G(i, k))
        :param i: station (0 to N-1)
        :param k: jobs in the network
        :return:
        """
        if k == 0:    # Bolch 8.12
            return 0.0
        else:         # Bolch 8.11
            val_list = [self.log_F(i, j) + self.GN_dict[i][k - j] for j in range(1, k + 1)]
            max_val = max(val_list)
            log_s = max_val + np.log(sum([np.exp(x - max_val) for x in val_list]))
            max_log = max(log_s, self.G_dict[self.numStations-1][k])
            if self.G_dict[self.numStations-1][k] - log_s <= 0.0:
                return -np.inf
            else:
                return max_log + np.log(np.exp(self.G_dict[self.numStations-1][k] - max_log) - np.exp(log_s - max_log))

    @lru_cache()
    def _GN(self, i, k):
        """
        :param i: station (0 to N-1)
        :param k: jobs in the network
        :return:
        """
        if k == 0:    # Bolch 8.12
            return 1.0
        else:         # Bolch 8.11
            return self.G(self.numStations - 1, k) - sum([self.F(i, j) * self.GN(i, k - j) for j in range(1, k + 1)])

    @lru_cache()
    def prob(self, i, k):
        return self._log_prob(i, k) if self.use_logs else self._prob(i, k)

    @lru_cache()
    def _log_prob(self, i, k):   # Bolch 8.7
        """
        prob k users in station i
        :param i:
        :param k:
        return self.F(i, k) * self.GN(i, self.numJobs - k) / self.G(self.numStations-1, self.numJobs)
        """
        return np.exp(self.log_F(i, k) + self.GN_dict[i][self.numJobs - k] - self.G_dict[self.numStations - 1][self.numJobs])

    @lru_cache()
    def _prob(self, i, k):   # Bolch 8.7
        """
        prob k users in station i
        :param i:
        :param k:
        return self.F(i, k) * self.GN(i, self.numJobs - k) / self.G(self.numStations-1, self.numJobs)
        """
        return self.F(i, k) * self._GN(i, self.numJobs - k) / self._G(self.numStations - 1, self.numJobs)

    @lru_cache()
    def tput(self, i):   # Bolch 8.14
        """
        station i throughput
        :param i: station
        return self.e[i] * self.G(self.numStations-1, self.numJobs-1) / self.G(self.numStations-1, self.numJobs)
        """
        return np.exp(np.log(self.e[i]) + self.G_dict[self.numStations-1][self.numJobs-1] - self.G_dict[self.numStations-1][self.numJobs])

    @lru_cache()
    def avg_busy_servers(self, i):
        """
        avg number of servers busy
        :param i: station
        :return: 
        """
        lwr = min(self.m[i], self.numJobs)
        return sum([k * self.probs_dict[i][k] for k in range(lwr)]) + self.m[i] * sum([self.probs_dict[i][k] for k in range(self.m[i], self.numJobs + 1)])

    @lru_cache()
    def rho(self, i):     # Bolch 7.21
        """
        prob station i is busy (utilization)
        :param i: station
        :return:
        """
        return self.tput(i) / (self.m[i] * self.mu[i])

    @lru_cache()
    def avg_jobs(self, i):  # Bolch 7.26
        """
        avg number of jobs in station i
        :param i: station
        :return:
        """
        return sum([k * self.probs_dict[i][k] for k in range(self.numJobs + 1)])

    @lru_cache()
    def avg_jobs_waiting(self, i):
        """
        avg number of jobs waiting in station i for service
        :param i: station
        :return: 
        """
        return 0.0 if self.numJobs <= self.m[i] else sum([(k - self.m[i]) * self.probs_dict[i][k] for k in range(1 + self.m[i], 1 + self.numJobs)])
        # return self.avg_jobs(i) - self.avg_busy_servers(i)

    @lru_cache()
    def avg_waiting_time(self, i):
        """
        avg time waiting for svc in queue i
        :param i: station 
        :return: 
        """
        return self.avg_response_time(i) - 1.0 / self.mu[i]

    @lru_cache()
    def avg_response_time(self, i):    # Bolch 7.30
        """
        station i response time
        :param i: station
        :return:
        """
        return self.avg_jobs(i) / self.tput(i)

    @lru_cache()
    def cycleTime(self):
        """
        job cycle time
        :return:
        """
        return sum([self.avg_response_time(i) * self.e[i] for i in range(self.numStations)])

if __name__ == "__main__":
    cfg_file = sys.argv[1]
    with open(cfg_file, 'r') as fp:
        arg_dict = json.load(fp)

    e = arg_dict['visit_ratios']
    m = arg_dict['number_of_servers']
    mu = arg_dict['service_rates']
    jobs = arg_dict['jobs']

    # example with an MR queue
    e = [1, 1]      # queue 0: think station, queue 1 processing station
    # r = cpuTime / totalTime = (1/mu)/(1/lbda + 1/mu). Set mu(cpu) = 1
    for r in [0.1, 0.5, 0.9]:
        for jobs in range(1, 100, 10):
            m = [jobs, 1]
            mu = [r / (1.0 - r), 1]
            q_ntw = BCMP('MR', e, m, mu, jobs, use_logs=True, iter=True)
            print('cpuTime/dbTime: ' + str(r) + ' jobs: ' + str(jobs) +
                  ' resp time: ' + str(q_ntw.cycleTime()) +
                  ' run queue: ' + str(q_ntw.avg_jobs_waiting(1)) +
                  ' cpu util: ' + str(int(100.0 * q_ntw.avg_busy_servers(1) / float(m[1]))) + '%' +
                  ' jobs in IO: ' + str(int(100.0 * q_ntw.avg_busy_servers(0) / float(m[0]))) + '%')
