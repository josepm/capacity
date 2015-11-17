"""
hyperbolic approximation to machine repair model
a = m^2 (1+2r) /(r^2(1+r)^2)
n0 = mr/(1+r)
q0 = m (r^2-r-1) / (r (1+r))
q(N)=q0+sqrt(A + (N-n0)^2) for N>m
q(N)=N r/(1+r) for N<=m

"""

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


class MR_hyperbolic:
    def __init__(self, lmbda, mu, servers, jobs):
        """
        :param r: r parameter = service time / think time (lambda/mu)
        :param m: servers
        """
        self.servers = servers
        self.r = lmbda / np.float(mu)
        self.a = servers**2 * (1.0 + 2.0 * self.r) / (self.r**2 * (1.0 + self.r)**2)
        self.n0 = servers * self.r/(1.0 + self.r)
        self.q0 = servers * (self.r**2 - self.r - 1.0) / (self.r * (1.0 + self.r))
        self.avg_queue = self.q0 + np.sqrt(self.a + (jobs - self.n0)**2) if jobs > self.servers else jobs * self.r / (1.0 + self.r)


if __name__ == '__main__':
    arg_dict = au.get_pars(sys.argv[1:])
    servers, think_time, processing_time = arg_dict['servers'], arg_dict['think_time'], arg_dict['processing_time']
    s_out = ''
    for s in servers:
        for p in processing_time:
            for jobs in range(1, 4 * s + 1):
                MR = MR_hyperbolic(1.0 / think_time, 1.0 / p, s, jobs)
                d = {'servers': s, 'r': MR.r}
                tput = (jobs - MR.avg_queue) / np.float(think_time)
                pwr = MR.r * (float(jobs) / MR.avg_queue - 1.0)
                d['jobs'] = jobs
                d['queue'] = MR.avg_queue
                d['pwr'] = pwr
                d['tput'] = tput
                s_out += json.dumps(d) + '\n'
    with open('/tmp/hyperbolic.json', 'w') as f:
        f.write(s_out)


