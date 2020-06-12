"""
Erlang Queues:
   Erlang-A
     M/M/m/Infinity + M-abandons with infinite buffer, based on http://ie.technion.ac.il/serveng/References/MMNG_thesis.pdf
     See https://airbnb.quip.com/rGZnAxWBOTNv/A-Queueing-Capacity-Model-for-Voice
     SLAs:
        Fraction poorly served or not served: P(W > t && !ABN) + P(ABN)
   Erlang-B: M/M/m/0 queue
      SLA: fraction not served = blocking probability
   Erlang-C: M/M/m/Infinity queue
      SLA: fraction poorly served = prob of waiting
   Erlang-A-PS
     M/M/m/Infinity + M-abandons with infinite buffer and processor sharing
     See https://airbnb.quip.com/jIzjA1bqSmjH/Queueing-Capacity-Model-for-Messaging
     SLAs:
        Fraction poorly served or not served: P(W > t && !ABN) + P(ABN)

Model Parameters
# lambda: arrival rate
# mu: service rate
# m: servers
# theta: abandonment rate
# K: max concurrenrcy
# q: context switching parameter
"""

import numpy as np
from functools import lru_cache

from capacity_planning.queueing.erlang import erlang_utils as e_ut
from capacity_planning.utilities import sys_utils as s_ut

class Erlang(object):
    def __init__(self, lbda, mu, m, verbose):
        self.lbda = lbda      # arrival rate
        self.mu = mu          # service rate
        self.m = max(1, int(np.round(m)))           # number of servers
        self.a = self.lbda / self.mu   # offered traffic
        self.verbose = verbose
        self.pars_ok = True

    def __str__(self):
        return ' lbda: ' + str(self.lbda) + ' mu: ' + str(self.mu) + ' m: ' + str(self.m)

    @lru_cache(maxsize=None)
    def erlB(self, use_log=False):
        """
        prob blocking in M/M/n/n
        see http://blog.wolfram.com/2013/03/21/the-mathematics-of-queues/
        erlB = a^m * e^(-a) / Gamma(1 + m, a) with Gamma(m, a) = Integral_a^inf e^(-t) t^(m-1) dt
        :param use_log: True, return the log of prob instead
        :return:
        """
        def log_erlB(a_val, m):
            """
            log of prob blocking in M/M/n/n
            :param a_val: offered traffic
            :param m: number of servers
            :return:
            """
            return m * np.log(a_val) - a_val - e_ut.log_gamma_upr(1 + m, a_val)

        if use_log is False:
            return np.exp(log_erlB(self.a, self.m))
        else:
            return log_erlB(self.a, self.m)

    @staticmethod
    @lru_cache(maxsize=None)
    def log_tail_sum(nx, dx):
        return nx - dx * np.log(nx) + e_ut.log_gammainc(1 + dx, lwr=0, upr=nx)

    def queue_ok(self, string=None):
        # check pars before run
        if self.pars_ok is False:
            s_ut.my_print('cannot execute. invalid parameters ' + str(string))
            return None


class ErlangA(Erlang):
    # https://ecommons.usask.ca/bitstream/handle/10388/etd-06222010-103338/ZhidongZhangThesisFinal.pdf
    # see http://ie.technion.ac.il/serveng/References/MMNG_thesis.pdf
    def __init__(self, lbda, mu, m, theta, verbose=False):
        super().__init__(lbda, mu, m, verbose)
        self.theta = theta               # abandonment rate (exp)
        self.sa = self.theta / self.mu   # avg service to abn abn ratio
        self.b = self.lbda / self.theta if self.theta > 0 else np.inf
        self.c = self.m * self.mu / self.theta if self.theta > 0 else np.inf
        if self.m < 1 or self.lbda <= 0 or self.mu <= 0 or self.theta < 0:
            s_ut.my_print('ErlangA: WARNING: invalid parameters: ' + self.__str__())
            self.pars_ok = False

    def __str__(self):
        return super().__str__() + ' theta: ' + str(self.theta)

    @lru_cache(maxsize=None)
    def m_fact(self):   # m^m/m!
        return self.m * np.log(self.m) - e_ut.log_gamma(1 + self.m)  # m^m / m!

    @lru_cache(maxsize=None)
    def _A(self, c, b, use_log=True):
        la = np.log(c) - c * np.log(b) + b + e_ut.log_gamma_lwr(c, b)
        if use_log == True:
            return la
        else:
            return np.exp(la)

    @lru_cache(maxsize=None)
    def prob_n(self, n, use_log=False):
        self.queue_ok(string=self.__str__())
        if n == 0:
            lp = -self.partition(use_log=True)
        elif 0 < n <= self.m:
            lp = n * np.log(self.a) - e_ut.log_gamma(n + 1) - self.partition(use_log=True)
        else:
            lp = self.prob_m(use_log=True) + (n - self.m) * np.log(self.b) + e_ut.log_gamma(1 + self.c) - e_ut.log_gamma(1 + self.c + n - self.m)
        if use_log == False:
            return np.exp(lp)
        else:
            return lp

    @lru_cache(maxsize=None)
    def partition(self, use_log=False):   # 1/p0
        lp = self.prob_m(use_log=True) - self.m_fact()
        if use_log == True:
            return -lp
        else:
            return np.exp(-lp)

    @lru_cache(maxsize=None)
    def prob_m(self, use_log=False):     # Prob(m) eq 5.5
        eb = self.erlB(use_log=True)
        la = self._A(self.c, self.b, use_log=True)
        ld = e_ut.log_trick([[0, 1], [eb + la, 1], [eb, -1]])[0]
        lp = eb - ld
        if use_log == True:
            return lp
        else:
            return np.exp(lp)

    @lru_cache(maxsize=None)
    def _gl(self, t):
        la = self._A(self.c, self.b, use_log=True)
        lgt = e_ut.log_gamma_lwr(self.c, self.b * np.exp(-self.theta * t))
        lg0 = e_ut.log_gamma_lwr(self.c, self.b)
        if np.isinf(lgt):
            if np.isinf(lg0):  # i.e. 0/0 in the natural scale: use asymptotics
                ln = self.c * np.log(self.b) - self.c * self.theta * t +\
                     e_ut.log_trick([[0, 1], [np.log(self.c), 1], [np.log(self.c) + np.log(self.b) - self.theta * t, -1]])[0]
                ld = np.log(self.c) + np.log(1 + self.c) + e_ut.log_gamma_lwr(self.c, self.b)
                return ln - ld
            else:  # num = 0 and den > 0 in the natural scale
                return -np.inf
        else:
            return la + lgt - lg0

    @lru_cache(maxsize=None)
    def sf(self, t, use_log=False):        # P(W>t) eq 5.18
        l = self.prob_m(use_log=True) + self._gl(t) - self.theta * t
        if use_log == True:
            return l
        else:
            return np.exp(l)

    @lru_cache(maxsize=None)
    def erlA(self, use_log=False):
        # P(W> 0)
        # Note lim_{theta -> inf} = erlB (0 patience) and lim_{theta -> 0} = erlC (infinite patience)
        return self.sf(0, use_log=use_log)


    @lru_cache(maxsize=None)
    def utilization(self):
        # server util: offered load to a single server
        leb = self.erlB(use_log=True)
        la = self._A(self.c, self.b, use_log=True)
        la1 = e_ut.log_trick([[la, 1], [0, -1]])
        lb1 = e_ut.log_trick([[0, 1], [leb, -1]])[0]
        lx1 = np.log(self.a) - np.log(self.m) + lb1 - leb
        lx = e_ut.log_trick([la1, [lx1, 1]])[0]
        return np.exp(self.prob_m(use_log=True) + lx)

    @lru_cache(maxsize=None)
    def poor_svc_prob(self, t, use_log=False):          # Prob(W>t & ~ABN) eq 5.20
        la = self._A(self.c, self.b * np.exp(-self.theta * t), use_log=True)
        lsf = self.sf(t, use_log=True)
        lr = np.log(self.a / self.m)
        lx = e_ut.log_trick([[0, 1], [-la, -1]])[0]
        ly = lsf + self.theta * t - lr + lx
        if use_log == True:
            return ly
        else:
            return np.exp(ly)

    @lru_cache(maxsize=None)
    def abn_prob(self, use_log=False):    # prob(abn) eq.5.23
        lw0 = self.sf(0, use_log=True)
        lps = self.poor_svc_prob(0, use_log=True)
        lt = e_ut.log_trick([[lw0, 1], [lps, -1]])[0]
        if use_log == True:
            return lt
        else:
            return np.exp(lt)

    @lru_cache(maxsize=None)
    def sla_prob(self, t, use_log=False):     # prob(W<= t & !ABN)
        lp = e_ut.log_trick([[0, 1], [self.abn_prob(use_log=True), -1], [self.poor_svc_prob(t, use_log=True), -1]])[0]
        try:
            if use_log == True:
                return lp
            else:
                return np.exp(lp)
        except RuntimeWarning:
            s_ut.my_print('Runtime warning::sla_prob::' + self.__str__() + 't: ' + str(t) + ' lp' + str(lp))
            if use_log == True:
                return lp
            else:
                return np.exp(lp)


class ErlangC(ErlangA):
    def __init__(self, lbda, mu, m, verbose=False):
        super().__init__(lbda, mu, m, 0.0, verbose)
        if self.m < 1 or self.lbda <= 0 or self.mu <= 0 or self.m <= self.a:
            s_ut.my_print('ErlangC: WARNING: invalid parameters: ' + self.__str__())
            self.pars_ok = False
        self.m_arr = np.array(self.m) if isinstance(self.m, (list, np.ndarray)) else self.m
        self.a_arr = np.array(self.a) if isinstance(self.a, (list, np.ndarray)) else self.a

    @lru_cache(maxsize=None)
    def erlC(self, use_log=False):
        # P(wait > 0) = sum_{k>=m} p(k) != p(Q=m)
        self.queue_ok(string=self.__str__())
        if self.m - self.a > 0:
            l_c = e_ut.log_trick([[0, 1], [-self.m * np.log(self.a) + self.a + np.log(self.m - self.a) +
                                           e_ut.log_gamma_upr(self.m, self.a), 1]])[0]
            if use_log is False:
                return np.exp(-l_c)
            else:
                return -l_c
        else:
            if use_log is False:
                return 1.0
            else:
                return 0.0

    @lru_cache(maxsize=None)
    def avg_wait(self):
        # avg wait time
        self.queue_ok(string=self.__str__())
        return (self.mmc_R() - 1.0) / self.mu

    @lru_cache(maxsize=None)
    def prob_m(self, use_log=False):
        # all servers busy
        self.queue_ok(string=self.__str__())
        return self.prob_n(self.m, use_log=use_log)

    @lru_cache(maxsize=None)
    def partition(self, use_log=False):        # return 1/p(0)
        self.queue_ok(string=self.__str__())
        l_p0 = np.log(self.m - self.a) + e_ut.log_gamma(self.m) - self.m * np.log(self.a) + self.erlC(use_log=True)  # log(p(0))
        if use_log is True:
            return -l_p0
        else:
            return np.exp(-l_p0)        # 1/p(0)

    @lru_cache(maxsize=None)
    def prob_n(self, n, use_log=False):
        self.queue_ok(string=self.__str__())
        p0 = 1.0 / self.partition(use_log=use_log) if use_log is False else -self.partition(use_log=use_log)
        if n == 0:
            return p0
        else:
            lp = n * np.log(self.a) - e_ut.log_gamma(1 + n)
            if n > self.m:
                lp += self.m_fact()
            if use_log is False:
                return p0 * np.exp(lp)
            else:
                return p0 + lp

    @lru_cache(maxsize=None)
    def avg_in_svc(self):    # number of avg busy servers
        return self.utilization() * self.m

    def utilization(self):
        # return self.lbda / (self.m * self.mu)
        return self.a / self.m

    @lru_cache(maxsize=None)
    def sys_avg(self):
        # avg number in system (wait + queue)
        l1 = self.erlC(use_log=True) + np.log(self.a) - np.log(self.m - self.a)
        return self.a + np.exp(l1)

    @lru_cache(maxsize=None)
    def mmc_Q(self, conditioned=False):
        """
        avg queue in the M/M/m queue, EXCLUDING SERVERS
        E[Q] = bz_srvs / (m - bz_srvs) * pC(m, bz_srvs) = avg queue length (excluding servers)
        E[Q | W > 0] = bz_srvs / (m - bz_srvs) = avg queue length (excluding servers) when there is waiting  ??????
        :param conditioned: if true compute E[Q | W > 0] otherwise, compute E[Q]
        :return: queue length v[i, j] = mmc_Q(n_servers_i, bz_srvs_j)
        """
        self.queue_ok(string=self.__str__())
        return (self.a_arr / (self.m_arr - self.a_arr)) * (1.0 if conditioned is True else self.erlC(use_log=False))

    @lru_cache(maxsize=None)
    def mmc_N(self, conditioned=False):
        """
        E[N] = avg number in the M/M/m queue (wait + svc)
        Waiting time (in queue): E[W] = E[Q]/lambda
        E[N] = lambda * (1/mu + E[W]) = bz_srvs + bz_srvs / (m - bz_srvs) * pC(m, bz_srvs)
        E[N|W>0] = m + bz_srvs / (m - bz_srvs) * pC(m, bz_srvs)
        :param conditioned: if true compute E[Q | W > 0] otherwise, compute E[Q]
        :return: queue length,  v[i, j] = mmc_N(n_servers_i, bz_srvs_j)
        """
        self.queue_ok(string=self.__str__())
        v_arr = self.a_arr if conditioned is False else self.m_arr
        q = np.where(self.a_arr >= self.m_arr, np.inf * np.ones(len(self.a_arr)), v_arr + self.erlC(use_log=False) * self.a_arr / (self.m_arr - self.a_arr))
        if len(q) == 1:
            return q[0]
        else:
            return q

    @lru_cache(maxsize=None)
    def mmc_R(self):
        """
        E[R] = avg response time in the M/M/m queue (wait + service) MEASURED IN SERVICE TIME UNITS.
        The actual response time is E[R]/mu
        Waiting time (in queue): E[W] = E[Q]/lambda
        E[R] = (1 + C(m, a) / (m-a)), a<m and a = lambda / mu
        :return: avg response time (wait + service) v[i, j] = mmc_R(n_servers_i, bz_srvs_j)
        """
        self.queue_ok(string=self.__str__())
        if self.m_arr > self.a_arr:
            return 1.0 + self.erlC(use_log=False) / (self.m_arr - self.a_arr)
        else:
            return np.inf

    @lru_cache(maxsize=None)
    def mmc_pwr(self, alpha=1.0):
        """
        M/M/c pwr function with parameters lambda, mu and m (# servers)
        a = lambda / mu = avg number of busy servers
        Q = number in queue. E[Q] = C(m, a) * a / (m-a)
        N = number in system. E[N] = a + E[Q]
        R = avg time in the system = avg_svc_time + avg_queueing_time = (1/mu) + E[Q]/lambda = (1/mu) * (1 + C(m, a) / (m - a))
        t = traffic = avg number in system = E[N] = X * R = a + C(m, a) * a / (m - a)
        P(a) = X(a) / R(a) = lambda / ((1/mu) * (1 + C(m, a) / (m - a))) = mu^2 * a / (1 + C(m, a) / (m - a))
        :param alpha: exponent on tput
        :return: P(t) =  lmbda^(1+alpha)/ q(m, a)
        """
        self.queue_ok(string=self.__str__())
        res = np.where(self.a >= self.m, 0.0 * np.ones(len(self.a)), (self.lbda ** (alpha + 1.0)) / self.mmc_Q()).ravel()
        if len(res) == 1:
            return res[0]
        else:
            return res

    @lru_cache(maxsize=None)
    def wait_SF(self, t, use_log=False):
        """
        P(wait > t)
        :param t: time
        :param use_log:
        :return:
        """
        self.queue_ok(string=self.__str__())
        if self.m - self.a > 0:
            l_pm = self.erlC(use_log=True)
            l_p = l_pm - self.mu * (self.m - self.a) * t
            if use_log is True:
                return l_p
            else:
                return np.exp(l_p)
        else:
            if use_log is False:
                return 1.0
            else:
                return 0.0

    @lru_cache(maxsize=None)
    def sla_prob(self, t, use_log=False):
        # prob(W <= t)
        self.queue_ok(string=self.__str__())
        lw = e_ut.log_trick([[0, 1], [self.wait_SF(t, use_log=True), -1]])[0]   # 1 - p(W>t) = 1 - p3 exp(-alpha t)
        if use_log is False:
            return np.exp(lw)
        else:
            return lw


class ErlangB(Erlang):
    def __init__(self, lbda, mu, m, verbose=False):
        super().__init__(lbda, mu, m, verbose)
        self.a = self.lbda / self.mu   # offered traffic

    @lru_cache(maxsize=None)
    def sla_prob(self):
        self.queue_ok(string=self.__str__())
        return 1 - self.erlB(use_log=False)

    def utilization(self):
        return self.lbda * self.sla_prob() / (self.m * self.mu)

    @lru_cache(maxsize=None)
    def erlB_retry(self, retry=0.0, eps=1E-6):
        """
        erlang B with retries
        r = bz_srvs
        a = r * erlB(n_servers, r)
        r = bz_srvs + p_retry * a
        :param retry: prob of retry for a blocked arrival
        :param eps: relative offered load error
        :return:
        """
        self.queue_ok(string=self.__str__())
        old_r, r, b = self.a, self.a, None
        delta = eps
        while delta >= eps:
            b = self.erlB(use_log=False)    # prob of a call gets blocked
            a1 = r * b                      # number of blocked calls
            r = self.a + retry * a1         # updated offered load = offered load + number of blocked calls that retry
            delta = (r - old_r) / old_r     # error on the offered load
            old_r = r
        return b * (1.0 - retry) / (1.0 - b * retry)  # prob loosing a call when each blocked call retries with prob retry

    @lru_cache(maxsize=None)
    def erlB_bulk(self, bsz):
        """
        Erlang B with bulk arrivals
        :param bsz: avg bulk size
        :return: prob of blocking
        """
        self.queue_ok(string=self.__str__())
        if self.a == 0:
            return 0.0
        if bsz <= 1:
            return self.erlB(use_log=False)

        q = (bsz - 1.0) / np.float(bsz)
        probs = np.zeros(1 + self.m, dtype=float)
        psum = 0.0
        bprod = 1.0
        probs[0] = 1.0 / np.float(self.m)
        for k in range(1, 1 + self.m):
            probs[k] = probs[k - 1] * (self.a * (1.0 - q) + q * (k - 1)) / np.float(k)
            psum += probs[k]
            bprod *= ((1.0 - q) * self.a + q * k) / np.float(k)
        p0 = 1.0 - np.float(self.m) * psum
        return p0 * bprod

