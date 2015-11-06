import numpy as np
import sys


"""
################################
# Implements Erlang B formulas
################################

erlB(servers, traffic)
servers(traffic, prob)
traffic(servers, prob)

p(m, rho) = (rho^m/m!)/Sum[rho^i/i!, {i, 0, m}]
rho = a = XR
X: xtions/sec
R = resp time (in secs)

################################

Algorithm
  assume rho >= 0 and m >= 0 and m integer
  return 0 if rho == 0
  s = 0
  for i = 1 to m
    s = (1 + s) * (i/rho)
  end
return 1 / (1 + s)
"""


def erlB(m, rho):
    """
    # m = # servers, rho = avg # of active connections
    :param m: number of servers
    :param rho: avg number of active requests (connections, threads, ...)
    :return: prob of blocking
    """
    if rho == 0:
        return 0
    if isinstance(m, int) and rho > 0:
        a = rho.to_f
        s = 0.0
        for n in range(1, m + 1):
            s = (n / a) * (1.0 + s)
        return 1.0 / (1.0 + s)
    else:   # interpolsate
        mceil = np.ceil(m).astype(int)
        mfloor = np.floor(m).astype(int)
        return erlB(mfloor, rho) + (m - mfloor) * (erlB(mceil, rho) - erlB(mfloor, rho))


def servers(rho, pblock):
    """
    compute the smallest # of servers m for given rho and blocking probability pblock such that erlB(m, rho) < pblock < erlB(m-1, rho)
    p(m, rho) decreases with m
    Given rho and a blocking prob B, find the largest m such that p(m) <= B
    bh < pblock and bl > pblock and h > l (the high number h provides the lower bound and vice-versa)
    :param rho: offered traffic
    :param pblock: blocking prob
    :return:  number of servers
    """

    if pblock > 1.0 or pblock < 0.0 or rho < 0.0:
        print 'Invalid parameters. prob: ' + str(pblock) + ' load: ' + str(rho)
        sys.exit(0)

    if pblock == 1.0:
        return 0

    if pblock == 0.0:
        return np.nan

#   bounds for m
    m_min = 16384       # lower bound: 2^14
    b = erlB(m_min, rho)
    while b <= pblock:
        m_min = np.floor(m_min / 2.0)
        b = erlB(m_min, rho)
        if m_min == 1:
            break
    m_max = 2 * m_min   # upper bound

#   do it the dumb way (no bissection)
    for m in range(m_min, m_max):
        b = erlB(m, rho)
        if b <= pblock:
            b1 = b
            b2 = erlB(m - 1, rho)
            return m - (pblock - b1) / (b2 - b1)
    print 'Invalid range'    # in case we ever get here
    sys.exit(0)


# increases with servers and pblock
# computes a
def traffic(servers, pblock):
    """
    computes the max traffic given servers and pblock
    :param servers:
    :param pblock:
    :return:
    """
    if pblock <= 0 or servers <= 0 or pblock >= 1:
      print 'Invalid parameters. prob: ' + str(pblock) + ' servers: ' + str(servers)
      sys.exit(0)

#   find a starting point that is reasonable
    a =  servers * np.exp(np.log(pblock) / float(servers))
    b = erlB(servers, a)
    u, l = 0, 0
    if b > pblock:
        u = a   # we found an upper bound
        while b > pblock:
          a /= 2.0
          b = erlB(servers, a)
        l = a
    else:      # we found a lower bound
        while b <= pblock:
          a *= 2.0
          b = erlB(servers, a)
        u = a

    iter, m = 0, 0
    while u - l > 1.0e-6 and iter < 500:
        iter += 1
        m = (u + l) / 2.0
        b = erlB(servers, m)
        if b > pblock:   # m is an upper bound
            u = m
        else:
            l = m
    return m


def erlBBulk(m, rho, bsz):
    """
    # m = # servers, rho = avg # of active connections
    :param m: number of servers
    :param rho: avg number of active requests (connections, threads, ...)
    :param bsz: avg bulk size
    :return: prob of blocking
    """
    if rho == 0:
        return 0
    if bsz <= 1:
      return erlB(m, rho)

    q = (bsz - 1.0) / np.float(bsz)
    probs = np.zeros(m + 1, dtype=float)
    psum = 0.0
    bprod = 1.0
    probs[0] = 1.0 / np.float(m)
    for k in range(1, m + 1):
        probs[k] = probs[k-1] * (rho * (1.0 - q) + q * (k - 1)) / np.float(k)
        psum += probs[k]
        bprod *= ((1.0 - q) * rho + q * k) / np.float(k)
    p0 = 1.0 - np.float(m) * psum
    return p0 * bprod


#
# ################################
# # Implements Erlang C formulas
# ################################
# # M/M/m/N queue model
# # r = lambda / mu
# # m: servers
# # nmax = total buffer (servers + queue). If nil, infinite buffer.
# # r < m when nmax = nil
# #
# # p(m, r, k) = p0 (r^k/k!)                if k <= m
# # p(m, r, k) = p0 (m^m/m!) (r/m)^(k - m)  if k >= m
# # p0 = 1/q0
# # q0(m, N) = Sum_{k=0}^{m-1} (r^k/k!) + (r^m / m!) * (1-(r / m)^(nmax + 1 - m)) /(1-(r / m))
# #
# #
# ################################
#
#
# def erlC(m, r)
#   return probTail(m, r, m, nil)
# end
#
# def probDensity(m, r, k, nmax = nil)  # M/M/m queue prob(=k)
#   return -1 if nmax == nil and r >= m
#   return probTail(m, r, k, nmax) - probTail(m, r, k + 1, nmax)
# end
#
# def probTail(m, r, n, nmax = nil)   # tail distribution: prob(>= n)
#   return -1 if nmax == nil and r >= m
#   p_0 = p0(m, r, nmax)
#   return probTail0(m, r, n, p_0, nmax)
# end
#
# def probTail0(m, r, n, p_0, nmax = nil)   # tail distribution: prob(>= n)
#   if n < m
#     sum, x = 1.0, 1.0
#     (1 .. n).each do |k|  # sum from 1 to n -1
#       x *= r / k.to_f     # r^k / k!
#       sum += x            # sum_{k=0}^{n - 1} r^k / k!
#     end
#     return 1.0 - p_0 * sum
#   else
#     rm = r / m.to_f
#     t = ((m ** m) / fact(m)) * (rm ** n) / (1.0 - rm)
#     t *= (1.0 - rm ** (nmax + 1 - n)) unless nmax == nil
#     return p_0 * t
#   end
# end
#
# def prob(m, r, k, nmax = nil)
#   return p0(m, r, nmax) * pL(r, k)  if k < m
#   return p0(m, r, nmax) * pH(r, k)  if k >= m
# end
#
# def pL(r, k)   #     r^k / k!  k <= m
#   return 1 if k == 0
#   return (r / k) * pL(r, k - 1)
# end
#
# def pH(r, k)    # r^k  k >= m
#   return 1 if k == 0
#   return r * pH(r, k - 1)
# end
#
# def p0(m, r, nmax = nil)
#   if r <= m or nmax != nil
#     return 1.0 / q0(m, r, nmax)
#   else
#     return 0
#   end
