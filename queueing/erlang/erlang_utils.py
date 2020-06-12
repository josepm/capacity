# ##########################################################
# #################### SPECIAL FUNCTIONS ###################
# ##########################################################

import os
import types
import scipy.special as sps
import numpy as np
from scipy.optimize import minimize_scalar
from copy import copy
from scipy.special import logsumexp as lsexp
import mpmath as mpm    # Scipy has problems with large args with gammaln and gammincc: we use the (slower but accurate) mpmath versions)
from functools import lru_cache

from capacity_planning.utilities import sys_utils as s_ut


# ###############################################################
# auxiliary functions
# ###############################################################

@lru_cache(maxsize=None)
def log_A(x, y):
    # x > 0, y >= 0
    return np.log(x) + y - x * np.log(y) + log_gamma_lwr(x, y)


@lru_cache(maxsize=None)
def log_expa(a, m):
    # eq. 6.77: a^(1-m) e^a Gamma[m,a] -> sum to m - 1
    return a + (1 - m) * np.log(a) + log_gamma_upr(m, a)


@lru_cache(maxsize=None)
def log_Jt(lbda, mu, m, theta, t):
    # should be decreasing with t
    q = m * mu / theta
    z = (lbda / theta) * np.exp(- theta * t)
    return lbda / theta - np.log(theta) + q * np.log(theta / lbda) + log_gammainc(q, lwr=0, upr=z, n=12)


@lru_cache(maxsize=None)
def log_JH(lbda, mu, m, theta, t):
    ly1 = log_Jt(lbda, mu, m, theta, t) - np.log(theta)
    ly2 = log_Jt(lbda, mu + theta / m, m, theta, t) - np.log(theta)
    return log_trick([[ly1, 1], [ly2, -1]])[0]


# #########################################################################
# ####################### mpmath wrappers #################################
# #########################################################################


def log_gammainc(a, lwr=0, upr=np.inf, n=24):
    prec_def, dps_def = mpm.mp.prec, mpm.mp.dps
    x_mpm = None
    while mpm.mp.dps >= 2:
        try:
            x_mpm = mpm.gammainc(np.float64(a), np.float64(lwr), np.float64(upr))
            break
        except:  #ValueError:
            x_mpm = None
            s_ut.my_print('pid: ' + str(os.getpid()) + ' WARNING:::erlang_utils::log_gammainc could not converge:: a: ' +
                  str(a) + ' lwr: ' + str(lwr) + ' upr: ' + str(upr) + ' dps: ' + str(mpm.mp.dps))
            if mpm.mp.dps == 15:  mpm.mp.dps = 16  # reset dps for easier divisions
            mpm.mp.dps = int(mpm.mp.dps / 2)

    if x_mpm is None:
        if np.isinf(upr):  # upr incomplete case
            if a / lwr <= 1:   # lwr = infinity
                x_mpm = mpm.mpf(-lwr + (a - 1) * np.log(lwr))
            else:  # a infinity
                v = [[np.abs((a - 0.5) * np.log(a) - a + np.log(2 * np.pi) / 2), 1], [np.abs(-lwr + a * np.log(lwr) - np.log(a)), -1]]
                lval, sgn = log_trick(v)
                x_mpm = mpm.mpf(lval * sgn)
        else:  # lwr incomplete case
            if a / upr > 1:   # a = infinity
                x_mpm = mpm.mpf(a * np.log(lwr) - np.log(a))
            else:  # -E^-z z^(-1 + a) + Gamma[a, 0]
                v = [[np.abs(-upr + (a - 1) * np.log(upr)), -1], [np.abs(log_gamma(a)), 1]]
                lval, sgn = log_trick(v)
                x_mpm = mpm.mpf(lval * sgn)

    x_str = mpm.nstr(x_mpm, n=n)
    v_str = x_str.split('e')
    try:
        l_m = np.log(np.float(v_str[0]))
        e = np.log(10) * np.float(v_str[1]) if len(v_str) == 2 else 0.0
        mpm.mp.prec, mpm.mp.dps = prec_def, dps_def  # reset
        return l_m + e
    except: # ValueError:
        s_ut.my_print('pid: ' + str(os.getpid()) + ' ERROR:::erlang_utils::log_gammainc:: a: ' + str(a) + ' lwr: ' + str(lwr) + ' upr: ' + str(upr) + ' dps: ' + str(mpm.mp.dps))
        return None


expi = np.frompyfunc(lambda x: float(mpm.ei(x)), 1, 1)

# #########################################################################
# ########## Framework for recursions and log sums ########################
# #########################################################################


def trampoline(gen, *args, **kwargs):
    g = gen(*args, **kwargs)
    while isinstance(g, types.GeneratorType):
        g = next(g)
    return g


def log_trick(arr):
    # 1. if Y = a1 + a2 + ... , then the log_trick input is an array of the form arr = [[log(|a1|), sign(a1)], [log(|a2|), sign(a2)]], ] ...
    # 3. returns [log(|Y|), sign(Y)]
    if isinstance(arr, list) or isinstance(arr, np.ndarray) or isinstance(arr, tuple):
        if len(arr) > 0:
            try:
                vals = [x[0] for x in arr]
                b = [x[1] for x in arr]
                return list(lsexp(vals, b=b, return_sign=True))
            except TypeError:
                s_ut.my_print('log_trick: invalid input format: ' + str(arr))
                return None
        else:  # empty array
            return [-np.inf, 0]
    else:      # if not list, assume float
        try:
            varr = float(arr)
            sgn = np.sign(varr)
            if sgn == 0:
                return [-np.inf, 0]
            else:
                return [np.log(sgn * varr), sgn]
        except ValueError:
            s_ut.my_print('log_trick::invalid input::' + str(arr))
            return None

# #########################################################################
# ### Argument extensions for gamma functions and exponential integrals ###
# #########################################################################


def log_gamma(a):
    return sps.gammaln(a)


def log_gamma_upr(a, z):
    # log of the upper incomplete gamma function for pos and neg args in the a.
    # z must be >= 0.
    # not normalized like in scipy
    if z == 0:
        return sps.gammaln(a)
    else:
        # s_ut.my_print('pid: ' + str(os.getpid()) + ' erlang_utils::log_gamma_upr:: a: ' + str(a) + ' z: ' + str(z))
        lg = trampoline(g_log_gamma_upr, a, z, acc_s=list(), acc_p=0, sgn=-1)
        return lg


def gamma_upr(a, z):
    # incomplete gamma function for pos and neg args in the a.
    # z must be >= 0.
    # not normalized like in scipy
    # Recursion: Gamma(a, z) = (Gamma(a+1, z) - e^(-z) * z^a)) / a
    return np.exp(log_gamma_upr(a, z))


def gamma_lwr_series(a, z, err=1.0e-6, n_max=1000):
    # https://www.maths.lancs.ac.uk/jameson/gammainc.pdf
    def n_term(a_, z_, n_):
        s = n_ + a_
        l = s * np.log(z_) - log_gamma(1 + n_) - np.log(s)
        if n_ % 2 == 0:
            return np.exp(l)
        else:
            return -np.exp(l)
    n, val = 2, n_term(a, z, 0) + n_term(a, z, 1)
    while n < n_max:
        n_val = val + n_term(a, z, n) + n_term(a, z, n + 1)
        if np.abs(n_val - val) < err:
            return n_val
        else:
            val = n_val
            n += 2
    # s_ut.my_print('WARNING: gama_lwr_series did not converge:: err: ' + str(err))
    return 0.0


def log_gamma_lwr(a, z):
    # log of the lower incomplete gamma function for pos and neg args in the a.
    # z must be >= 0.
    # not normalized like in scipy
    # log(Gamma(a)) + log(1 - Gamma(a, z)/Gamma(a))
    if z == 0.0:
        return -np.inf
    else:
        d = log_gamma_upr(a, z) - sps.gammaln(a)
        if np.abs(d) <= 1.0e-12:
            g = gamma_lwr_series(a, z)
            if g > 0.0:
                return np.log(g)
            else:
                return -np.inf
        else:
            arr = [[0, 1], [log_gamma_upr(a, z) - sps.gammaln(a), -1]]
            l_arr = log_trick(arr)
            return sps.gammaln(a) + l_arr[0]


def gamma_lwr(a, z):
    # lower incomplete gamma function for any a and z >= 0
    return np.exp(log_gamma_lwr(a, z))


def g_log_gamma_upr(a, z, acc_s=list(), acc_p=0, sgn=-1):
    # generator to compute the log_gamma_upr
    # z must be >= 0.
    # not normalized like in scipy
    # Recursion for negative a: Gamma(a, z) = (Gamma(a+1, z) - e^(-z) * z^a)) / a
    # call: trampoline(g_log_gamma_upr, a, z)
    if a > 0:
        # s_ut.my_print('erlang_utils::a: ' + str(a) + ' z: ' + str(z) + ' acc_p: ' + str(acc_p))
        lg = log_gammainc(a, lwr=z, upr=np.inf)  # not normalized
        vz = lg - acc_p
        l_v = log_trick([[vz, np.sign(-sgn)], log_trick(acc_s)])
        if l_v[0] > sps.gammaln(a):  # round off error
            l_v[0] = sps.gammaln(a)
        yield l_v[0]
    else:
        if z == 0:
            yield np.inf
        if a == 0:
            vz = np.log(np.abs(expi(-z))) - acc_p
            l_v = log_trick([[vz, np.sign(-sgn)], log_trick(acc_s)])
            yield l_v[0]
        else:
            acc_p += np.log(-a)
            x = -sgn * np.exp(a * np.log(z) - z - acc_p)
            acc_s.append([np.log(np.abs(x)), np.sign(x)])
            yield g_log_gamma_upr(a + 1, z, acc_s=copy(acc_s), acc_p=acc_p, sgn=-sgn)


def expn(n, z):
    # expn function any n,
    # z > 0
    # int_1^inf e^(-x t) / t^n dt = z^(n-1) Gamma(1-n, x)
    return np.exp(log_expn(n, z))


def log_expn(n, z):
    # log of expn function any n
    # z > 0
    # log(int_1^inf e^(-x t) / t^n dt) = (n - 1) * log(z) + Log(Gamma(1-n, x))
    l_g = trampoline(g_log_gamma_upr, 1 - n, z, acc_s=list(), acc_p=0, sgn=-1)
    return (n - 1) * np.log(z) + l_g


@lru_cache(maxsize=None)
def partitions(bins, n_sum):
    # Return the list of all integers (n1, .., n_bins) such that n1, ..., n_bins >= 0 and n_1 + ...+ n_bins = n_sum
    # P(bins - 1, n_sum) = sum_{s=1}^n_sum P(bins - 1, s).append(n_sum - s)
    if bins == 1:
        return [[n_sum]]
    else:
        return [[v for v in p] + [n_sum - s] for s in range(0, n_sum + 1) for p in partitions(bins-1, s)]

# ##########################################################
# ##########################################################
# ##########################################################
# ############### Objective Functions ######################
# ##########################################################
# ##########################################################
# ##########################################################


# ##########################################################
# ##########################################################
# ##########################################################
# ############### Generic Erlang Functions #################
# ##########################################################
# ##########################################################
# ##########################################################


def servers(bz_srvs, p_tgt, erl_func, use_log=False, ctr=0, mult=2.0, verbose=True):
    """
    compute the smallest # of servers m for given bz_srvs and target probability (blocking or queueing) p_tgt, eg min{m: erl_funcB(m, bz_srvs) < p_tgt < erl_func(m-1, bz_srvs)}
    return max{m: erl_func(m, bz_srvs) <= p_tgt}
    :param bz_srvs: offered traffic. Numeric
    :param p_tgt: target prob. Either blocking or waiting prob. Numeric. If use_log is True, prob is a log_prob
    :param erl_func: Erlang function to use (erlB or erlC), This defines the underlying queueing model.
    :param use_log: True/False (tells if the prob is in logs or not)
    :param ctr: recursion ctr
    :param mult: upper bound multiplier
    :param verbose: s_ut.my_print errors
    :return: number of servers. Numeric
    Test: erl_func(servers(bz_srvs, p_tgt, erl_func), bz_srvs) = p_tgt
    """
    def func(m_val, *pars):
        a, lp, e_func = pars
        return (lp - e_func(m_val, a, use_log=True)) ** 2

    prob = np.exp(p_tgt) if use_log is True else p_tgt
    l_prob = np.log(p_tgt) if use_log is False else p_tgt
    ret = err_checks('servers()', n_svrs=1.0, prob=prob, l_prob=l_prob, bz_srvs=bz_srvs, verbose=verbose)  # check prob range
    if ret is None:
        return np.nan

    l_bnd = bz_srvs if 'erlC' in erl_func.__name__ else 1.0
    u_bnd, p0 = l_bnd + 1.0, 1.0
    while p0 > prob:
        p0 = erl_func(u_bnd, bz_srvs)
        u_bnd *= mult

    res = minimize_scalar(func, args=(bz_srvs, l_prob, erl_func), method='bounded', bounds=(l_bnd, u_bnd))
    return gen_return(res, u_bnd, servers, [bz_srvs, p_tgt, erl_func, use_log, ctr, mult])


def traffic(n_servers, p_tgt, erl_func, use_log=False, ctr=0, mult=2.0, verbose=True):
    """
    computes the max busy servers such that erl_func(n_servers, traffic) = p_tgt
    :param n_servers: number of servers available. Numeric
    :param p_tgt: target prob. Numeric. prob or log_prob
    :param erl_func: Erlang function to use (erlB or erlC)
    :param use_log: True/False (tells if the prob is in logs or not)
    :param ctr: recursion ctr
    :param mult: upper bound multiplier
    :param verbose: s_ut.my_print errors
    :return: traffic that meets p_tgt and n_servers
    Test: erl_func(n_servers, traffic(n_servers, p_tgt, erl_func)) = p_tgt
    """
    def func(a_val, *pars):
        m_val, lp, e_func = pars
        return (lp - e_func(m_val, a_val, use_log=True)) ** 2

    prob = np.exp(p_tgt) if use_log is True else p_tgt
    l_prob = np.log(p_tgt) if use_log is False else p_tgt
    ret = err_checks('traffic()', n_svrs=1.0, prob=prob, l_prob=l_prob, bz_srvs=1.0, verbose=verbose)  # check prob range
    if ret is None:
        return np.nan

    if 'erlC' in erl_func.__name__:
        u_bnd = n_servers
    else:
        u_bnd, p0 = n_servers, 0.0
        while p0 < prob:
            p0 = erl_func(n_servers, u_bnd)
            u_bnd *= mult
    res = minimize_scalar(func, args=(n_servers, l_prob, erl_func), method='bounded', bounds=(0.0, u_bnd))
    return gen_return(res, u_bnd, traffic, [n_servers, p_tgt, erl_func, use_log, ctr, mult])


def gen_return(res, u_bnd, g_func, pars):
    x, p, e_func, u_log, ctr, mult = pars
    if res.status == 0:
        if np.abs(res.x - u_bnd) > 1.0e-3:
            return res.x
        else:      # result is the upper bound: try again with higher u_bnd
            if ctr < 4:
                return g_func(x, p, e_func, use_log=u_log, ctr=ctr+1, mult=mult*2)
            else:
                return None
    else:
        return None


def servers_bisect(bz_srvs, p_tgt, erl_func, ctr_max=100000, eps=1e-6, use_log=False, srv_mult=1000000, verbose=True):
    """
    Bi-section is FASTER than direct minimization but requires more parameters
    compute the smallest # of servers m for given bz_srvs and target probability (blocking or queueing) p_tgt, eg min{m: erl_funcB(m, bz_srvs) < p_tgt < erl_func(m-1, bz_srvs)}
    return max{m: erl_func(m, bz_srvs) <= p_tgt}
    NOTE: m is at most bz_srvs / p_tgt
    :param bz_srvs: offered traffic. Numeric
    :param p_tgt: target prob. Either blocking or waiting prob. Numeric. If use_log is True, prob is a log_prob
    :param erl_func: Erlang function to use (erlB or erlC), This defines the underlying queueing model.
    :param ctr_max: max iterations for upper servers bound
    :param eps: error bound
    :param use_log: True/False (tells if the prob is in logs or not)
    :param srv_mult: max number of servers multiplier
    :param verbose: s_ut.my_print errors
    :return: number of servers. Numeric
    Test: erl_func(servers(bz_srvs, p_tgt, erl_func), bz_srvs) = p_tgt
    """

    # bisect approach
    def same_sign(aval, bval):
        return bool(aval * bval > 0.0)

    def f_val(lp, func, svrs, a):
        return func(svrs, a, use_log=True) - lp

    prob = np.exp(p_tgt) if use_log is True else p_tgt
    l_prob = np.log(p_tgt) if use_log is False else p_tgt
    ret = err_checks('servers()', n_svrs=1.0, prob=prob, l_prob=l_prob, bz_srvs=bz_srvs, verbose=verbose)  # check prob range
    if ret is None:
        return np.nan

    # bounds for m
    m_min, ctr, ret_val = 1.0, 0, np.nan
    m_max = min(int(np.ceil(bz_srvs / prob)) if prob > 0.0 else srv_mult * bz_srvs, srv_mult * bz_srvs)
    while ctr < ctr_max:
        mid = np.ceil((m_min + m_max) / 2.0)
        old_mid, old_min, old_max = mid, m_min, m_max
        mid_val = f_val(l_prob, erl_func, mid, bz_srvs)
        if np.abs(mid_val) < eps:
            ret_val = mid
        if same_sign(f_val(l_prob, erl_func, m_min, bz_srvs), mid_val):
            m_min = mid
        else:
            m_max = mid
        if m_min == old_min and m_max == old_max and mid == old_mid:
            vm = [m_min, m_max, mid]
            errs = [np.abs(f_val(l_prob, erl_func, m, bz_srvs)) for m in vm]
            return vm[errs.index(min(errs))]
        ctr += 1

    return ret_val


def traffic_bisect(n_servers, p_tgt, erl_func, use_log=False, srv_mult=1000000.0, ctr_max=500, eps=1e-6, verbose=True):
    """
    Bi-section is FASTER than direct minimization but requires more parameters
    computes the max busy servers such that erl_func(n_servers, traffic) = p_tgt
    :param n_servers: number of servers available. Numeric
    :param p_tgt: target prob. Numeric. prob or log_prob
    :param erl_func: Erlang function to use (erlB or erlC)
    :param ctr_max: max iterations for upper servers bound
    :param eps: error bound
    :param use_log: True/False (tells if the prob is in logs or not)
    :param srv_mult: max number of servers multiplier
    :param verbose: s_ut.my_print errors
    :return:
    Test: erl_func(n_servers, traffic(n_servers, p_tgt, erl_func)) = p_tgt
    """
    prob = np.exp(p_tgt) if use_log is True else p_tgt
    l_prob = np.log(p_tgt) if use_log is False else p_tgt
    ret = err_checks('traffic()', n_svrs=1.0, prob=prob, l_prob=l_prob, bz_srvs=1.0, verbose=verbose)  # check prob range
    if ret is None:
        return np.nan

    cnt, ret_val = 0, np.nan
    u, l = min(n_servers / prob, srv_mult * n_servers), 1.0
    if u < l:
        l, u = u, l

    while u - l > eps and cnt < ctr_max:
        cnt += 1
        ret_val = (u + l) / 2.0
        p_val = erl_func(n_servers, ret_val, use_log=True)
        if p_val > p_tgt:   # ret_val is an upper bound
            u = ret_val
        else:
            l = ret_val
    return ret_val


def err_checks(f_name, n_svrs=None, bz_srvs=None, prob=None, l_prob=None, verbose=True):  # generic bounds checker
    if prob == 0.0 and np.isneginf(l_prob):
        if verbose:
            s_ut.my_print(f_name + ': Invalid parameters. prob: ' + str(prob))
        return None
    if prob >= 1.0 or l_prob > 0.0:
        if verbose:
            s_ut.my_print(f_name + ': Invalid parameters. prob: ' + str(prob))
        return None
    if n_svrs <= 0:
        if verbose:
            s_ut.my_print(f_name + ': Invalid parameters. servers: ' + str(n_svrs))
        return None
    if bz_srvs <= 0:
        if verbose:
            s_ut.my_print(f_name + ': Invalid parameters. bz servers: ' + str(bz_srvs))
        return None
    return True
