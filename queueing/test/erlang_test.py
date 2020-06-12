"""
unit tests for erlang_queues.py
$ python erlang_test.py
"""

import unittest
import numpy.testing as npt
import numpy as np

from capacity_planning.queueing import erlang_queues


class TestErlangFunctions(unittest.TestCase):
    def test_erlB(self):
        # values based on mma results
        servers = [100, 10000, 100, 10000, 100]
        traffic = [120, 12000, 20, 200, 10000]
        log_probs = [-1.62826407041, -1.78927734649, -84.1661482002, -29325.7541713, -0.01004931575]
        z_vals = zip(servers, traffic, log_probs)
        for z in z_vals:
            m, a, lp = z
            npt.assert_almost_equal(lp, erlang_queues.ErlangB(m, a, use_log=True), decimal=6, err_msg='erlB test failure for m: ' + str(m) + ' a: ' + str(a), verbose=True)

    def test_erlC(self):
        # values based on mma results
        servers = [100, 10000, 100, 10000, 100, 10000, 10000]
        traffic = [90, 9000, 10, 100, 50, 5000, 9990]
        results = [-1.52813224460, -56.8266885378, -143.375505741, -36157.2159266, 0.0, -59093.0759064, -15.4613286611]
        z_vals = zip(servers, traffic, results)
        for z in z_vals:
            m, a, lp = z
            lmbda, mu = a, 1
            npt.assert_almost_equal(lp, erlang_queues.ErlangC(lmbda, mu, m).erlC(use_log=True),
                                    decimal=6, err_msg='erlC test failure for m: ' + str(m) + ' a: ' + str(a) + ' log_prob: ' + str(lp), verbose=True)

    def test_servers(self):
        # values based on mma results
        # servers(bz_srvs, p_tgt, erl_func, ctr_max=100, eps=1e-12, use_log=False, verbose=False)
        err = 1e-12
        l_err = int(np.abs(np.log10(err)) / 4)
        # erlB
        servers = [100, 10000, 100, 10000]
        traffic = [120, 12000, 20, 200]
        log_probs = [-1.62826407041, -1.78927734649, -84.1661482002, -29325.7541713]
        z_vals = zip(traffic, log_probs, servers)
        for z in z_vals:
            a, lp, m = z
            npt.assert_almost_equal(m, erlang_queues.servers(a, lp, erlang_queues.ErlangB, use_log=True), decimal=l_err,
                                    err_msg='servers-erlB test failure for m: ' + str(m) + ' a: ' + str(a) + ' log_prob: ' + str(lp), verbose=True)

        # erlC
        servers = [100, 10000, 100, 10000]
        traffic = [90, 9000, 10, 100]
        results = [-1.52813224460, -56.8266885378, -143.375505741, -36157.2159266]
        z_vals = zip(servers, traffic, results)
        for z in z_vals:
            m, a, lp = z
            npt.assert_almost_equal(m, erlang_queues.servers(a, lp, erlang_queues.ErlangC, use_log=True), decimal=l_err,
                                    err_msg='servers-erlC test failure for m: ' + str(m) + ' a: ' + str(a) + ' log_prob: ' + str(lp), verbose=True)

    def test_traffic(self):
        # values based on mma results
        # bz_servers(n_servers, p_tgt, erl_func, ctr_max=500, eps=1e-6, use_log=False)
        # erlB
        err = 1e-12
        l_err = int(np.abs(np.log10(err)) / 4)
        servers = [100, 10000, 100, 10000]
        traffic = [120, 12000, 20, 200]
        log_probs = [-1.62826407041, -1.78927734649, -84.1661482002, -29325.7541713]
        z_vals = zip(traffic, log_probs, servers)
        for z in z_vals:
            a, lp, m = z
            npt.assert_almost_equal(a, erlang_queues.traffic(m, lp, erlang_queues.ErlangB, use_log=True), decimal=l_err,
                                    err_msg='traffic-erlB test failure for m: ' + str(m) + ' a: ' + str(a) + ' log_prob: ' + str(lp), verbose=True)

        # erlC
        servers = [100, 10000, 100, 10000]
        traffic = [90, 9000, 10, 100]
        results = [-1.52813224460, -56.8266885378, -143.375505741, -36157.2159266]
        z_vals = zip(servers, traffic, results)
        for z in z_vals:
            m, a, lp = z
            npt.assert_almost_equal(a, erlang_queues.traffic(m, lp, erlang_queues.ErlangC, use_log=True), decimal=l_err,
                                    err_msg='traffic-erlC test failure for m: ' + str(m) + ' a: ' + str(a) + ' log_prob: ' + str(lp), verbose=True)


if __name__ == '__main__':
    unittest.main()


