"""
SLA vs weekly work hours
Shows how to compute the SLA and prob(ABN) from work-hours in two examples.
ignore shrinkage and occupancy
Usage: $ python SLA-example.py
"""

from capacity_planning.queueing import erlang as ERL

wwh = 40000                      # week work hours
hbat = 20                        # tix hbat in minutes
agents = wwh / 40                # agents available (40 hrs per agent per week)
tix = wwh * 60 / hbat            # ttl tix per week
s_agent = 5                      # shifts per agent (5 * 8)
agent_shifts = s_agent * agents  # ttl agent shifts available
abn_mult = 1                    # abandonment multiplier: a customer will wait abn_mult * hbat on average before abandoning

# uniform case: all hours/days identical (3 shifts per day) -- worst case
lbda = tix / (7 * 24 * 60)       # input rate in tix per min
ttl_shifts = 7 * 3               # ttl shifts during the week
m = agent_shifts / ttl_shifts    # agents per shift

q_obj = ERL.ErlangA(lbda, 1 / hbat, m, 1/(abn_mult * hbat))
pSLA = q_obj.sla_prob(2)         # pWait <= 2 min & !ABN
pABN = q_obj.abn_prob()          # prob abandonment
print('uniform case::agents per shift: ' + str(int(m)) + ' p(Wait <= 2min & !ABN) = ' + str(round(pSLA, 2)) + ' p(ABN) = ' + str(round(pABN, 2)))

# Non-uniform case (example): all tix come only during business hours: one shift per day
lbda = tix / (7 * 8 * 60)        # input rate in tix per min
ttl_shifts = 7                   # ttl shifts during the week
m = agent_shifts / ttl_shifts    # agents per shift

q_obj = ERL.ErlangA(lbda, 1 / hbat, m, 1/(abn_mult * hbat))
pSLA = q_obj.sla_prob(2)         # pWait > 2 mina & !ABN
pABN = q_obj.abn_prob()          # prob abandonment
print('non-uniform case::agents per shift: ' + str(int(m)) + ' p(Wait <= 2min & !ABN) = ' + str(round(pSLA, 2)) + ' p(ABN) = ' + str(round(pABN, 2)))



