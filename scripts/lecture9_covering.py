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
# # Lecture 9: Set Covering and Shift Scheduling
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture9_covering.ipynb)

# %% [markdown]
# This notebook covers the set cover and covering problems, and their main practical use,
# shift scheduling: choosing the cheapest set of shifts (or facilities) that between them
# cover every point that needs covering (an hour, a location, ...). Unlike the
# [transportation problem](lecture9_transportation.ipynb), integrality is essential here.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - formulate set cover and covering problems as ILO models;
# - formulate shift-scheduling problems as covering problems and solve them with pulp.

# %%
import pulp

# %% [markdown]
# ## Set Cover and Covering Problems
#
# In the set cover problem we have a universe $U = \{1, \dots, m\}$ and sets
# $S_1, \dots, S_n$ with $S_i \subseteq U$, and we want the smallest selection of sets whose
# union is all of $U$. The classic motivation is facility location: let $U$ be the incident
# locations in a region and each $S_i$ the locations reachable within a target
# response time from candidate base station $i$; then set cover asks for the fewest base
# stations (ambulances, fire stations, ...) that cover the whole region. (IBM has used it
# for efficient virus scanning: scan for a small covering collection of overlapping
# signature sets rather than every signature.)
#
# With binary $x_i$ (1 = select set $i$) and $a_{ui} = 1$ if $u \in S_i$:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i=1}^{n} x_i \\
# \text{subject to} \quad & \sum_{i=1}^{n} a_{ui} x_i \ge 1 \text{ for all } u \in U; \\
# & x_i \in \{0, 1\} \text{ for all } i.
# \end{aligned}
# $$
#
# > **Erratum applied (p. 98):** the summation index is $x_j$, not $x_i$ (the book prints
# > $\sum_{j=1}^{n} a_{uj} x_i$).
#
# The integrality constraint is essential here. Take $U = \{1, 2, 3\}$ with
# $S_1 = \{1, 2\}$, $S_2 = \{1, 3\}$, $S_3 = \{2, 3\}$. No single set covers $U$, so the
# integer optimum is 2 (any two sets). But the LO relaxation can set every
# $x_i = \tfrac12$: each element is then covered by $\tfrac12 + \tfrac12 = 1$, at total
# "cost" $1.5$. Adding the three constraints gives $2(x_1 + x_2 + x_3) \ge 3$, so the
# relaxation can never beat $1.5$, and rounding the $0.5$'s up gives all three sets, which
# is worse than the true optimum of 2.
#
# A slightly larger instance, solved with pulp:

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
    set_cover += (
        pulp.lpSum(pick[s] for s in covers if u in covers[s]) >= 1,
        f"cover_{u}",
    )

solver = pulp.getSolver("COIN_CMD", msg=False)

set_cover.solve(solver)
print("stations:", [s for s in covers if pick[s].value() == 1])

# %% [markdown]
# The covering problem generalizes set cover: each element $u$ must be covered $b_u$
# times, a set may be chosen more than once ($x_i \in \{0, 1, 2, \dots\}$), and each set $i$
# has a cost $c_i$:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i=1}^{n} c_i x_i \\
# \text{subject to} \quad & \sum_{i=1}^{n} a_{ui} x_i \ge b_u \text{ for all } u \in U; \\
# & x_i \in \{0, 1, 2, \dots\} \text{ for all } i.
# \end{aligned}
# $$
#
# (shift-scheduling)=
# ### Shift Scheduling
#
# The covering problem's main use is shift scheduling (studied by Dantzig in 1954 for
# toll-booth staffing): split the day into time intervals $U$, let each shift type $i$ be
# the set $S_i$ of intervals it works, $b_u$ the required staffing in interval $u$, $c_i$
# the cost of one worker on shift $i$, and $x_i$ the number of workers assigned that shift.
# The required staffing per interval typically comes from a forecast (predictive
# analytics).
#
# A small shop is open 8:00–12:00 (four one-hour intervals) and needs $b = (3, 6, 7, 4)$
# staff. Four shift types are available:

# %%
shift_intervals = {
    "morning": {1, 2},
    "midday": {2, 3},
    "late": {3, 4},
    "full": {1, 2, 3, 4},
}
shift_cost = {"morning": 30, "midday": 30, "late": 30, "full": 50}
required = {1: 3, 2: 6, 3: 7, 4: 4}

roster = pulp.LpProblem(name="shift_scheduling", sense=pulp.LpMinimize)
count = {
    s: pulp.LpVariable(name=s, lowBound=0, cat="Integer")
    for s in shift_intervals
}
roster += pulp.lpSum(shift_cost[s] * count[s] for s in shift_intervals)
for u, need in required.items():
    roster += (
        pulp.lpSum(
            count[s] for s in shift_intervals if u in shift_intervals[s]
        )
        >= need
    )

solver = pulp.getSolver("COIN_CMD", msg=False)

roster.solve(solver)
print(
    "roster:",
    {s: count[s].value() for s in shift_intervals},
    "cost",
    roster.objective.value(),
)

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.5 "Example ILO Problems"
#   (covering). Spreadsheet material replaced with pulp.
# - Dantzig, G.B. (1954). "A comment on Edie's 'Traffic delays at toll booths.'" *Journal of
#   the Operations Research Society of America*, 2(3):339–341.
