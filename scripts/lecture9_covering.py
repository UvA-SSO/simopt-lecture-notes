# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# description: "Set cover and covering problems, and their main practical use, shift scheduling."
# thumbnail: null
# ---
# # Lecture 9: Set Covering and Shift Scheduling
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture9_covering.ipynb)

# %% [markdown]
# This notebook covers the set cover and covering problems, and their main practical use,
# shift scheduling: choosing the cheapest set of shifts (or facilities) that between them
# cover every point that needs covering (an hour, a location, ...). Unlike the
# [transportation problem](lecture9_transportation.ipynb), integrality is essential here.
# Both problems are introduced in the same steps as the transportation problem:
# practical motivation, generic model, the model for an example, its solution in pulp,
# and extensions.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - formulate set cover and covering problems as ILO models;
# - explain why the integrality constraints cannot be dropped in a set cover problem;
# - formulate shift-scheduling problems as covering problems and solve them with pulp.

# %%
import pulp

# %% [markdown]
# ## The Set Cover Problem
#
# ### Practical Motivation
#
# An ambulance service wants to open as few ambulance bases as possible, while every
# address in its region can still be reached from some base within the target response
# time (in the Netherlands, 15 minutes). For every candidate location we know which
# addresses it can reach in time; the question is which candidates to open. Fire
# stations, police posts and parcel lockers lead to the same question. A less obvious
# application comes from IBM, which used it for efficient virus scanning: instead of
# scanning for every known virus signature, scan for a small collection of code
# fragments that between them occur in all known viruses.

# %% [markdown]
# ### Modeling
#
# #### Problem Definition and Example
#
# We have a universe $U = \{1, \dots, m\}$ of elements that must be covered, and $n$ sets
# $S_1, \dots, S_n$ with $S_i \subseteq U$. The problem is to find the smallest
# collection of these sets whose union is all of $U$. In the ambulance application, $U$
# is the set of addresses and $S_i$ the set of addresses that candidate base $i$ reaches
# in time.
#
# As an example, a region has six incident locations, $U = \{1, \dots, 6\}$, and six
# candidate bases B1, ..., B6, which reach the following locations in time:
#
# | base | B1 | B2 | B3 | B4 | B5 | B6 |
# |---|---|---|---|---|---|---|
# | locations reached | 1, 2, 3 | 2, 3, 4 | 4, 5 | 5, 6 | 1, 6 | 3, 4, 5 |
#
# #### Decision Variables
#
# For every set we decide whether to take it or not, which is a yes/no decision. We
# model it with a binary decision variable per set:
#
# $$
# x_i =
# \begin{cases}
# 1 & \text{if set } S_i \text{ is taken,} \\
# 0 & \text{otherwise,}
# \end{cases}
# \quad i = 1, \dots, n.
# $$
#
# Which set contains which element is data, not a decision. To write it as numbers, let
# $a_{iu} = 1$ if $u \in S_i$ and $a_{iu} = 0$ otherwise.
#
# #### Objective
#
# We minimize the number of sets taken, $\sum_{i=1}^{n} x_i$.
#
# #### Constraints
#
# Every element $u$ must be in at least one of the sets taken. The number of sets taken
# that contain $u$ is $\sum_{i=1}^{n} a_{iu} x_i$, because only the sets with
# $a_{iu} = 1$ count. So, for every $u \in U$,
#
# $$
# \sum_{i=1}^{n} a_{iu} x_i \ge 1.
# $$
#
# We use "$\ge 1$" and not "$= 1$": an element may well be covered twice (two bases
# that both reach an address). Requiring exactly one would be a different, much more
# restrictive problem that often has no feasible solution at all. Finally,
# $x_i \in \{0, 1\}$.
#
# #### Complete ILO Model
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i=1}^{n} x_i \\
# \text{s.t.} \quad & \sum_{i=1}^{n} a_{iu} x_i \ge 1, \quad u \in U \\
# & x_i \in \{0, 1\}, \quad i = 1, \dots, n.
# \end{aligned}
# $$

# %% [markdown]
# ### Modeling the Example
#
# With $x_1, \dots, x_6$ for bases B1, ..., B6, each location gives one constraint, with
# the bases that reach it:
#
# $$
# \begin{aligned}
# \min \quad & x_1 + x_2 + x_3 + x_4 + x_5 + x_6 \\
# \text{s.t.} \quad & x_1 + x_5 \ge 1 & \text{(cover location 1)} \\
# & x_1 + x_2 \ge 1 & \text{(cover location 2)} \\
# & x_1 + x_2 + x_6 \ge 1 & \text{(cover location 3)} \\
# & x_2 + x_3 + x_6 \ge 1 & \text{(cover location 4)} \\
# & x_3 + x_4 + x_6 \ge 1 & \text{(cover location 5)} \\
# & x_4 + x_5 \ge 1 & \text{(cover location 6)} \\
# & x_i \in \{0, 1\}, \quad i = 1, \dots, 6.
# \end{aligned}
# $$

# %% [markdown]
# ### Solving the Example in pulp
#
# Two things are new in pulp. First, `cat="Binary"` makes a variable binary. Second, the
# data is stored as the sets themselves (a Python `set` per base) instead of the
# numbers $a_{iu}$: the sum $\sum_i a_{iu} x_i$ becomes a sum over only those bases whose
# set contains $u$, written with an `if` inside the generator.

# %%
covers = {
    "B1": {1, 2, 3},
    "B2": {2, 3, 4},
    "B3": {4, 5},
    "B4": {5, 6},
    "B5": {1, 6},
    "B6": {3, 4, 5},
}
universe = sorted(set().union(*covers.values()))

set_cover = pulp.LpProblem(name="set_cover", sense=pulp.LpMinimize)
pick = {s: pulp.LpVariable(name=s, cat="Binary") for s in covers}
set_cover += pulp.lpSum(pick.values())
for u in universe:
    n_covering = pulp.lpSum(pick[s] for s in covers if u in covers[s])
    set_cover += n_covering >= 1, f"cover_{u}"

set_cover.solve(pulp.PULP_CBC_CMD(msg=False))
print("status:", pulp.LpStatus[set_cover.status])
print("bases:", [s for s in covers if pick[s].value() == 1])

# %% [markdown]
# Three bases suffice. Two cannot: location 6 is only reached by B4 and B5. With B4, a
# second base would have to reach locations 1 to 4, and with B5 it would have to reach
# locations 2 to 5, and no base does either.

# %% [markdown]
# ### Extensions
#
# #### Are the Integrality Constraints Needed?
#
# In the transportation problem, we could drop integrality and still get whole numbers.
# Here we cannot. Take the small instance $U = \{1, 2, 3\}$ with $S_1 = \{1, 2\}$,
# $S_2 = \{1, 3\}$, $S_3 = \{2, 3\}$. No single set covers $U$, so the integer optimum is
# 2 (any two sets). But the LO relaxation, with $0 \le x_i \le 1$ instead of
# $x_i \in \{0, 1\}$, can set every $x_i = \tfrac12$: each element is then covered by
# $\tfrac12 + \tfrac12 = 1$, at total "cost" $1.5$. Adding the three constraints gives
# $2(x_1 + x_2 + x_3) \ge 3$, so the relaxation can never beat $1.5$, and rounding the
# $0.5$'s up gives all three sets, which is worse than the true optimum of 2.
#
# #### The Covering Problem
#
# The covering problem generalizes set cover in three ways: each element $u$ must be
# covered $b_u$ times instead of once, a set may be chosen more than once
# ($x_i \in \{0, 1, 2, \dots\}$), and each set $i$ has a cost $c_i$:
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i=1}^{n} c_i x_i \\
# \text{s.t.} \quad & \sum_{i=1}^{n} a_{iu} x_i \ge b_u, \quad u \in U \\
# & x_i \in \{0, 1, 2, \dots\}, \quad i = 1, \dots, n.
# \end{aligned}
# $$
#
# With $b_u = 1$ and $x_i$ binary, it is a set cover problem in which bases have
# different costs. Its main application, shift scheduling, is the next problem.

# %% [markdown]
# (shift-scheduling)=
# ## Shift Scheduling
#
# ### Practical Motivation
#
# A shop, a call center or a hospital ward needs a certain number of staff at work in
# every hour of the day, and that number varies over the day. Staff are hired for shifts
# of fixed types (a morning shift, a full-day shift, ...), each with its own cost. How
# many workers should be scheduled on each shift type, so that every hour has enough
# staff at the lowest total cost? The required staffing per hour typically comes from a
# demand forecast (predictive analytics). Dantzig studied this problem in 1954 for the
# staffing of toll booths.

# %% [markdown]
# ### Modeling
#
# #### Problem Definition and Example
#
# Shift scheduling is a covering problem. Split the day into time intervals
# $U = \{1, \dots, m\}$, and let $b_u$ be the required staffing in interval $u$. Each of
# the $n$ shift types $i$ works a set $S_i \subseteq U$ of intervals and costs $c_i$ per
# worker. As before, $a_{iu} = 1$ if shift type $i$ works in interval $u$ and $0$
# otherwise. The problem is to find the cheapest number of workers per shift type such
# that every interval has at least its required staffing.
#
# As an example, a small shop is open from 8:00 to 12:00 (four one-hour intervals) and
# has four shift types:
#
# | shift type | morning | midday | late | full |
# |---|---|---|---|---|
# | works | 8-10 | 9-11 | 10-12 | 8-12 |
# | intervals $S_i$ | 1, 2 | 2, 3 | 3, 4 | 1, 2, 3, 4 |
# | cost $c_i$ (€) | 30 | 30 | 30 | 50 |
#
# The required staffing is $b = (3, 6, 7, 4)$ for the intervals 8-9, 9-10, 10-11 and
# 11-12.
#
# #### Decision Variables
#
# Several workers can work the same shift type, so taking a set is no longer a yes/no
# decision but a number:
#
# $$
# x_i = \text{number of workers scheduled on shift type } i, \quad i = 1, \dots, n,
# $$
#
# with $x_i \in \{0, 1, 2, \dots\}$, since we cannot schedule half a worker.
#
# #### Objective
#
# We minimize the total staff cost, $\sum_{i=1}^{n} c_i x_i$.
#
# #### Constraints
#
# The number of workers present in interval $u$ is the sum of the workers on all shift
# types that work in $u$, which must be at least the requirement:
#
# $$
# \sum_{i=1}^{n} a_{iu} x_i \ge b_u, \quad u \in U.
# $$
#
# Overstaffing is allowed ("$\ge$"): with fixed shift lengths it is often unavoidable,
# since a worker needed in one interval may also be present in a quieter one.
#
# #### Complete ILO Model
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i=1}^{n} c_i x_i \\
# \text{s.t.} \quad & \sum_{i=1}^{n} a_{iu} x_i \ge b_u, \quad u \in U \\
# & x_i \in \{0, 1, 2, \dots\}, \quad i = 1, \dots, n.
# \end{aligned}
# $$
#
# This is exactly the covering problem from the previous section.

# %% [markdown]
# ### Modeling the Example
#
# With $x_1, x_2, x_3, x_4$ the number of morning, midday, late and full shifts:
#
# $$
# \begin{aligned}
# \min \quad & 30x_1 + 30x_2 + 30x_3 + 50x_4 \\
# \text{s.t.} \quad & x_1 + x_4 \ge 3 & \text{(8-9)} \\
# & x_1 + x_2 + x_4 \ge 6 & \text{(9-10)} \\
# & x_2 + x_3 + x_4 \ge 7 & \text{(10-11)} \\
# & x_3 + x_4 \ge 4 & \text{(11-12)} \\
# & x_i \in \{0, 1, 2, \dots\}, \quad i = 1, \dots, 4.
# \end{aligned}
# $$

# %% [markdown]
# ### Solving the Example in pulp
#
# The code has the same shape as the set cover code. The only change in the variables
# is `cat="Integer"` with `lowBound=0`, which gives $x_i \in \{0, 1, 2, \dots\}$, and
# the constraints now have the right-hand side `need` instead of 1.

# %%
shift_intervals = {
    "morning": {1, 2},
    "midday": {2, 3},
    "late": {3, 4},
    "full": {1, 2, 3, 4},
}
shift_cost = {"morning": 30, "midday": 30, "late": 30, "full": 50}
required = {1: 3, 2: 6, 3: 7, 4: 4}
shifts = list(shift_intervals)

roster = pulp.LpProblem(name="shift_scheduling", sense=pulp.LpMinimize)
count = {s: pulp.LpVariable(name=s, lowBound=0, cat="Integer") for s in shifts}
roster += pulp.lpSum(shift_cost[s] * count[s] for s in shifts)
for u, need in required.items():
    staffed = pulp.lpSum(count[s] for s in shifts if u in shift_intervals[s])
    roster += staffed >= need, f"staff_{u}"

roster.solve(pulp.PULP_CBC_CMD(msg=False))
print("status:", pulp.LpStatus[roster.status])
print("roster:", {s: count[s].value() for s in shifts})
print("total cost:", roster.objective.value())

# %% [markdown]
# The optimal roster has 3 midday, 1 late and 3 full shifts, at a cost of €270. Every
# requirement is met exactly, so here there is no overstaffing.

# %% [markdown]
# ### Extensions
#
# #### More Realistic Shifts
#
# Real instances have many more intervals (for example, half hours over a whole day)
# and many more shift types, with different lengths, start times and breaks. The
# [call-center exercise](lecture9_exercises.ipynb#ex-6-13) is such an instance.
#
# #### Not Meeting Every Requirement
#
# Requiring $b_u$ staff in every interval can be expensive when a single peak interval
# forces extra shifts. An alternative is to allow under- and overstaffing and penalize
# both in the objective. This needs the modeling trick for absolute values from
# [Robust Regression](lecture10_modeling-tools.ipynb) and is an exercise there.
#
# #### Interactions Between Intervals
#
# Here the required staffing per interval is given. In a call center, understaffing in
# one interval leaves a queue of waiting callers for the next one, so the requirements
# depend on the schedule itself. Then only simulation can evaluate a schedule; see
# [Simulation Optimization](lecture13_simulation-optimization.ipynb#eg-8-4).

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.5 "Example ILO Problems"
#   (covering). Spreadsheet material replaced with pulp.
