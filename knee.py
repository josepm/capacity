"""
A function that computes the knee(s) of an XY curve.
A curve knee signals the start of a new operating regime.
Assumes that Y is noisily increasing with X.
A curve knee is a very subjective concept.
The method is based on quantile regression.
This subjectivity is captured by the confidence level that a knee was found and the percentile used in the quantile regression.
"""

from statsmodels.regression.quantile_regression import QuantReg
import scipy.stats as sp
from operator import itemgetter
import numpy as np
import bisect
import sys


def find_ge_idx(a, x):  # Find the index of the leftmost item greater than or equal to x
    return bisect.bisect_left(a, x)  # if larger than len(a), not found


def check_prob(val, name):
    if val < 0.0 or val > 1.0:
        print 'invalid ' + name + ': ' + str(val)
        sys.exit(0)


def find_knee(X, Y, q=0.75, conf_level=0.999, q_init=0.5, n_knees=1):
    """
    Finds the knee of the XY curve (i.e. where Y shoots up in '"non-linear" fashion with respect to X)
    Assumes that Y is noisily increasing with X.
    The choice of q_init, q and conf_level reflects the subjectivity of the problem.
    - larger q_init will detect knees 'later' (i.e. for higher values of X or miss them altogether)
    - larger conf_level will detect knees 'later'
    - larger q will detect knees 'earlier'
    Example (M/M/1):
    X = np.random.uniform(low=0, high=1, size=100)
    Y = np.maximum(0, 1.0 / (1-X) + np.random.normal(0, 1, size=100))
    plt.scatter(X, Y)
    find_knee(X, Y, q=0.5, conf_level=0.999, q_init = 0.5)
    find_knee(X, Y, q=0.25, conf_level=0.999, q_init = 0.5)
    find_knee(X, Y, q=0.75, conf_level=0.999, q_init = 0.5)

    :param X: independent values (n x 1 list or np array)
    :param Y: dependent values (n x 1 list or np array)
    :param q: knee quantile level. The lower q, the less sensitive to knee detection, i.e. the knee, if any, will be detected at higher values of X.
    :param q_init: the percentile value where we start looking for the knee, e.g. if q_init = 0.5, we look for knees past the median of X.
    :param conf_level: knee detection confidence level. Set very high if we want knee certainty.
    :param n_knees: number of knees to detect
    :param knee_list: knee_list output
    :return: knee list
    """

    if len(X) != len(Y):
        print 'invalid input lengths. X: ' + str(len(X)) + ' Y: ' + str(len(Y))
        sys.exit(0)

    check_prob(q, 'q')
    check_prob(q_init, 'q_init')
    check_prob(conf_level, 'conf_level')
    if not(isinstance(n_knees, int)) or n_knees < 0:
        print 'invalid n_knees: ' + str(n_knees)
        sys.exit(0)

    # close recursion
    if n_knees == 0:
        return []

    # sort by increasing X and add 1's for the intercept
    x0 = np.ones(len(X))  # add 1's for intercept
    Z = zip(x0, X, Y)
    Z.sort(key=itemgetter(1))

    init_cnt = int(q_init * len(Z))
    Z_q, Z_k = Z[:init_cnt], Z[init_cnt:]
    X_q, Y_q = np.array([z[:-1] for z in Z_q]), np.array([z[-1] for z in Z_q])
    q_reg_obj = QuantReg(endog=Y_q, exog=X_q)
    mdl = q_reg_obj.fit(q=q)
    ones, X_k, Y_k = zip(*Z_k)             # already sorted!
    Y_preds = mdl.predict(zip(ones, X_k))  # predict all values from q-itle onwards
    signs = np.sign(Y_k - Y_preds)         # 1 if positive, -1 if negative, 0 if equal
    upr = np.maximum(0, signs)
    cum_upr = int((1.0 - q) * init_cnt) + np.cumsum(upr)  # cum_upr: count of points over regression line
    ttl_cnt = range(init_cnt, len(Z))                     # total running count
    rv = sp.binom(n=ttl_cnt, p=1.0 - q)
    diffs = 1.0 - conf_level - rv.sf(x=cum_upr - 1)
    knee_idx = find_ge_idx(diffs, 0.0)                    # knee: the first time we have binom_test(p_val) < 1-conf_level
    x_knee = X_k[knee_idx] if knee_idx < len(X_k) else None
    if x_knee is not None:
        if n_knees > 1:
            Z_n = [zn for zn in Z_k if zn[1] >= x_knee]
            if len(Z_n) > 10:
                ones, X_n, Y_n = zip(*Z_n)
                return [x_knee] + find_knee(X_n, Y_n, q=q, conf_level=conf_level, q_init=q_init, n_knees=n_knees - 1)
            else:
                return [x_knee]
        else:
            return [x_knee]
    else:
        return []
