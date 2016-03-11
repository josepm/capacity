"""
A function that computes the knee(s) of an XY curve.
A curve knee signals the start of a new operating regime.
Assumes that Y is noisily increasing with X.
A curve knee is a very subjective concept.
The method is based on quantile regression.
This subjectivity is captured by the confidence level that a knee was found and the percentile used in the quantile regression.
"""

from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.proportion import binom_test
from operator import itemgetter
import numpy as np


def find_knee(X, Y, q=0.5, conf_level=0.999, q_init=None, n_knee=1):
    """
    Finds the knee of the XY curve (i.e. where Y shoots up in '"non-linear" fashion with respect to X)
    Assumes that Y is noisily increasing with X.
    The choice of q and conf_level reflects the subjectivity of the problem
    Example (M/M/1):
    X = np.random.uniform(low=0, high=1, size=100)
    Y = 1.0 / (1-X)
    find_knee(X, Y, q=0.5, conf_level=0.999) -> knee at x = 0.49
    find_knee(X, Y, q=0.25, conf_level=0.999) -> knee at x = 0.798
    find_knee(X, Y, q=0.75, conf_level=0.999) -> knee at x = 0.26

    :param X: independent values (n x 1 list or np array)
    :param Y: dependent values (n x 1 list or np array)
    :param q: knee quantile level. The lower q, the less sensitive to knee detection
    :param q_init: the percentile value where we start looking for the knee.
                   If None, it looks for the most dense area between X values, which should be the low load region
    :param conf_level: knee detection confidence level. Set very high if we want knee certainty. At least 0.999, otherwise too many false knees
    :param n_knee: number of knees to detect
    :param knee_list: knee_list output
    :return: knee list
    """
    if n_knee == 0:
        return []
    x0 = np.ones(len(X))  # add 1's for intercept
    Z = zip(x0, X, Y)
    Z.sort(key=itemgetter(1))

    if q_init is None:
        dens = [(Z[idx][1], idx / (Z[idx][1] - Z[0][1])) for idx in range(1, len(Z))]
        x_start = max(dens, key=itemgetter(1))[0]
    else:
        x_start = Z[int(q_init * len(Z))][1]
    Z_q = [z for z in Z if z[1] <= x_start]
    Z_k = [z for z in Z if z[1] > x_start]
    Y_q = np.array([z[-1] for z in Z_q])
    X_q = np.array([z[:-1] for z in Z_q])  # add 1's for intercept
    q_reg_obj = QuantReg(endog=Y_q, exog=X_q)
    mdl = q_reg_obj.fit(q=q)

    upr_cnt, lwr_cnt = (1.0 - q) * len(Z), q * len(Z)
    for z in Z_k:
        if z[-1] > mdl.predict(z[:-1])[0]:
            upr_cnt += 1
        else:
            lwr_cnt += 1
        b = binom_test(upr_cnt, upr_cnt + lwr_cnt, prop=1.0 - q, alternative='larger')
        if b < 1.0 - conf_level:
            Z_n = [zn for zn in Z_k if zn[1] >= z[1]]
            if n_knee > 1 and len(Z_n) > 10:
                ones, X_n, Y_n = zip(*Z_n)
                return [z[1]] + find_knee(X_n, Y_n, q=q, conf_level=conf_level, q_init=q_init, n_knee=n_knee - 1)
            else:
                return [z[1]]
    return []





