"""
A simple staffing model
M/M/s + M
Arrival rate: lambda
Service rate: mu
Servers: s
Abandonment rate: theta
gamma = theta / mu = service time to abandonment time ratio, the inverse of the number of service times we wait till abandonment, i.e. theta = 0 means no abandonment

Staffing: s = a + beta * sqrt(a) with a = lbda / mu offered load
beta captures the SLA cost
SLA: P(Wait > w) <= q_SLA

Model approximation:
- when there is wait, the system behaves as an MM1 with parameters lambda and s * mu
- P(Wait > w) = P(Wait > 0) P_MM1(Wait > w | Wait > 0) where
    - the MM1 has parameters lambda, s * mu in the approx above
    - P(Wait > 0) = (1 + sqrt(gamma) h(beta / sqrt(gamma)) / h(-beta))^(-1) with h(x) = phi(x) / (1 + Phi(x)) N(0,1), hazard function (HW approximation to capture abandonments)
- When w is very high, we expect beta -> 0 and concurrency gains from messaging, : lambda -> lambda / m where m is the concurrency level.

Parameters:
    - interaction arrival rate: lambda
    - interaction processing rate: mu
    - svc time to abn time ratio: gamma
    - SLA P(Wait > w) <= q_SLA
        - SLA wait time: w
        - SLA fraction: q_SLA

Algorithm:
    - Solve the SLA equation for beta
    - agents = a + beta * sqrt(a) with a = lambda / mu

"""

import scipy.stats as sps
from scipy.optimize import minimize
import numpy as np
import pandas as pd


def n_hazard(x):
    return sps.norm.pdf(x) / sps.norm.sf(x)


def mm1_cwait(lbda, mu, t):
    # P(wait > t|wait > 0)
    return np.exp(-(mu - lbda) * t)


def prob_delay(beta, gamma):
    g_sqrt = np.sqrt(gamma)
    x = 1 + g_sqrt * n_hazard(beta / g_sqrt) / n_hazard(-beta)
    return 1 / x


def prob_wait(lbda, mu, gamma, beta, t):
    return prob_delay(beta, gamma) * mm1_cwait(lbda, mu, t)


def obj(_beta, *pars):
    lbda, mu, gamma, w, q = pars
    a = lbda / mu
    srvs = a + _beta * np.sqrt(a)
    return (q - prob_wait(lbda, srvs * mu, gamma, _beta, w)) ** 2


def get_agents(_lbda, _mu, _gamma, _w, _q_sla):
    res = minimize(obj, x0=1, method='Nelder-Mead', args=(_lbda, _mu, _gamma, _w, _q_sla))
    if res.success is True:
        beta = res.x[0]
        print(str(_w) + ' ' + str(beta))
        a = lbda / mu
        return a + beta * np.sqrt(a)
    else:
        print('no convergence: ' + str(res))
        return None


# ###############################
# parameters

mu = 1.0 / (15 * 60)    # service rate in 1/sec, 15 mins
lbda = 2                # interaction arrival rate in interactions / sec
gamma = 1 / 10000             # svc time to abn ratio
min_wait = 0            # min SLA wait in secs
max_wait = 36000        # max SLA wait in secs
n_pts = 100             # points between min and max wait
q_sla = 0.95             # SLA quantile

waits = np.linspace(min_wait, max_wait, num=n_pts)
agents = [get_agents(lbda, mu, gamma, w, q_sla) for w in waits]

df = pd.DataFrame({'wait_SLA': waits, 'agents': agents})
df.set_index('wait_SLA', inplace=True)
df.plot(grid=True, style='o-')



