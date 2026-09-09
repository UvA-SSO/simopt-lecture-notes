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
# # Lecture 9: Applications of (Integer) Linear Optimization

# %% [markdown]
# This notebook shows the breadth of problems that fit the (integer) linear optimization
# framework — the transportation problem, the set cover and covering problems, and shift
# scheduling — partly for their own sake and partly as inspiration for modeling your own
# problems. We start with *why* we lean so heavily on linear (and integer-linear) models in
# the first place.

# %%
import pulp

# %% [markdown]
# ## Why Linearity Matters, and What Integrality Buys Back
#
# Linearity is what makes LO efficiently solvable: the feasible region is a convex
# polyhedron, the optimum sits at a corner, and a local optimum is automatically global. As
# soon as the objective or a constraint is nonlinear, both of those break:
#
# - **Nonlinear objective.** Maximize $x_1 x_2$ subject to $x_1 + x_2 \le 1$,
#   $x_1, x_2 \ge 0$. The optimum is $(0.5, 0.5)$, in the *interior* of an edge, not at a
#   corner.
# - **Nonlinear constraint.** Maximize $x_1 + x_2$ subject to $\min(x_1, x_2) = 0$,
#   $x_1 \le 2$, $x_2 \le 1$. The feasible set is two line segments meeting at the origin
#   (either $x_1 = 0$ or $x_2 = 0$). It has two *local* optima, $(2, 0)$ and $(0, 1)$; you
#   cannot be sure which is global without checking both.
#
# Nonlinear optimization therefore needs slower, less reliable algorithms. But there is a
# large and useful middle ground: **integer** constraints. On the one hand, requiring
# $x_i \in \{0, 1, 2, \dots\}$ is itself a nonlinear constraint. On the other hand, many
# *other* nonlinearities — an either/or choice, a fixed cost that applies only when an
# activity is used, a "this constraint holds only if..." condition — can be expressed with
# integer (usually binary) variables and otherwise-linear constraints, and then solved with
# branch and bound. That is why so much modeling effort goes into casting a problem as ILO.
# [Machine Scheduling](lecture9_advanced-modeling.ipynb) is devoted to those tricks; this
# notebook is about problems that are ILO more directly.

# %% [markdown]
# ## The Transportation Problem
#
# We must ship a single good from $n$ **sources** (supply $a_i$ at source $i$) to $m$
# **destinations** (demand $b_j$ at destination $j$), at a cost $c_{ij}$ per unit shipped on
# link $i \to j$. How much should we ship on each link so that every demand is met, no
# supply is exceeded, and total cost is minimized?
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{i=1}^{n} \sum_{j=1}^{m} c_{ij} x_{ij} \\
# \text{subject to} \quad & \sum_{j=1}^{m} x_{ij} \le a_i \text{ for } i = 1, \dots, n; \\
# & \sum_{i=1}^{n} x_{ij} \ge b_j \text{ for } j = 1, \dots, m; \\
# & x_{ij} \ge 0 \text{ for all } i, j.
# \end{aligned}
# $$
#
# This is an LO problem (no integrality needed). If a link $i \to j$ does not exist, use a
# very large $c_{ij}$. Many assignment problems — staff to tasks, students to rooms — have
# this same shape.
#
# Two warehouses supply three stores:

# %%
supply = {"W1": 30, "W2": 25}
demand = {"S1": 20, "S2": 15, "S3": 20}
cost = {
    ("W1", "S1"): 4,
    ("W1", "S2"): 6,
    ("W1", "S3"): 8,
    ("W2", "S1"): 5,
    ("W2", "S2"): 3,
    ("W2", "S3"): 4,
}

transport = pulp.LpProblem(name="transportation", sense=pulp.LpMinimize)
ship = {(i, j): pulp.LpVariable(name=f"x_{i}_{j}", lowBound=0) for (i, j) in cost}

transport += pulp.lpSum(cost[i, j] * ship[i, j] for (i, j) in cost)
for i, cap in supply.items():
    transport += pulp.lpSum(ship[i, j] for j in demand) <= cap, f"supply_{i}"
for j, req in demand.items():
    transport += pulp.lpSum(ship[i, j] for i in supply) >= req, f"demand_{j}"

transport.solve(pulp.PULP_CBC_CMD(msg=False))
print("shipments:", {k: v.value() for k, v in ship.items() if v.value() > 0})
print("total cost:", transport.objective.value())

# %% [markdown]
# :::{exercise}
# :label: ex-6-8
#
# Solve the following transportation problem in pulp, where "x" means there is no
# connection (use a very large cost, e.g. `1e6`, in place of $\infty$): source 1 has supply
# 10 and costs $(0, 5, \text{x}, 0)$ to destinations 1–4; source 2 has supply 6 and costs
# $(4, 6, 4, 3)$; source 3 has supply 10 and costs $(2, 4, 4, 6)$; each destination has
# demand 5.
# :::

# %% [markdown]
# (transshipment-problem)=
# ### Transshipment
#
# If goods can pass through **intermediate nodes** on the way from sources to destinations,
# we have the *transshipment problem*. It is solved by adding, for every intermediate node
# $k$, a flow-conservation constraint — what comes in must go out:
#
# $$
# \sum_{i} x_{ik} = \sum_{j} x_{kj}.
# $$
#
# This extends to a full network by stacking several layers of intermediate nodes, and it
# is the model behind the shortest-path and maximum-flow problems in
# [Algorithms and Heuristics](lecture11_algorithms-heuristics.ipynb).

# %% [markdown]
# ## Set Cover and Covering Problems
#
# In the **set cover problem** we have a universe $U = \{1, \dots, m\}$ and sets
# $S_1, \dots, S_n$ with $S_i \subseteq U$, and we want the smallest selection of sets whose
# union is all of $U$. The classic motivation is **facility location**: let $U$ be the
# incident locations in a region and each $S_i$ the locations reachable within a target
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
# **The integrality constraint is essential here.** Take $U = \{1, 2, 3\}$ with
# $S_1 = \{1, 2\}$, $S_2 = \{1, 3\}$, $S_3 = \{2, 3\}$. No single set covers $U$, so the
# integer optimum is 2 (any two sets). But the LO relaxation can set every
# $x_i = \tfrac12$: each element is then covered by $\tfrac12 + \tfrac12 = 1$, at total
# "cost" $1.5$. Adding the three constraints gives $2(x_1 + x_2 + x_3) \ge 3$, so the
# relaxation can never beat $1.5$ — and rounding $0.5$'s up gives all three sets, which is
# worse than the true optimum of 2.
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
    set_cover += pulp.lpSum(pick[s] for s in covers if u in covers[s]) >= 1, f"cover_{u}"

set_cover.solve(pulp.PULP_CBC_CMD(msg=False))
print("stations:", [s for s in covers if pick[s].value() == 1])

# %% [markdown]
# The **covering problem** generalizes set cover: each element $u$ must be covered $b_u$
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
# The covering problem's main use is **shift scheduling** (studied by Dantzig in 1954 for
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
count = {s: pulp.LpVariable(name=s, lowBound=0, cat="Integer") for s in shift_intervals}
roster += pulp.lpSum(shift_cost[s] * count[s] for s in shift_intervals)
for u, need in required.items():
    roster += pulp.lpSum(count[s] for s in shift_intervals if u in shift_intervals[s]) >= need

roster.solve(pulp.PULP_CBC_CMD(msg=False))
print("roster:", {s: count[s].value() for s in shift_intervals}, "cost", roster.objective.value())

# %% [markdown]
# :::{exercise}
# :label: ex-6-12
#
# Solve the following ILO problem, inspired by Guenin, Könemann, and Tunçel (2014). A
# swimming pool is open for 12 hours and the lifeguards on duty must be selected. Each
# lifeguard has fixed working hours (first and last hour worked) and a wage. Select the
# cheapest set of lifeguards with at least one on duty every hour. (Ben + Celia + Fred is a
# feasible solution.)
#
# | Lifeguard | Ann | Ben | Celia | Dick | Estelle | Fred |
# |---|---|---|---|---|---|---|
# | Hours | 1–6 | 1–4 | 5–8 | 7–10 | 7–12 | 9–12 |
# | Wage | 8 | 6 | 6 | 3 | 7 | 3 |
#
# Use a matrix (nested list or dict) for $a_{ui}$.
# :::
#
# :::{exercise}
# :label: ex-6-13
#
# The required staffing in a call center, 9am–9pm in 30-minute intervals, is:
#
# 10, 11, 13, 16, 16, 13, 11, 10, 10, 11, 12, 13, 14, 14, 13, 11, 10, 9, 9, 10, 9, 8, 8, 8.
#
# Two shift types are available:
#
# - 8 hours working, with a 30-minute unpaid break in the middle, wage €20/hr, starting
#   every half hour from 9:00 to 12:30;
# - 4 consecutive hours, wage €24/hr, starting every half hour from 9:00 to 17:00.
#
# Formulate this as a covering problem and solve it with pulp.
# :::
#
# :::{exercise}
# :label: ex-6-14
#
# For one school day, classes must be assigned to professors so that each class has one hour
# with each professor it needs and no professor teaches two classes at once. A matrix
# $a_{cp}$ gives which classes need which professors ($a_{cp} = 1$ if class $c$ needs
# professor $p$). Formulate an ILO model that minimizes the total number of hours classes
# spend at school (a class stays until right after its last professor hour).
# :::
#
# :::{exercise}
# :label: ex-6-15
#
# Pairs must be made among $n$ students; each student lists the students they are willing to
# work with. Formulate an ILO model maximizing the number of pairs, each student in at most
# one pair (not everyone need be paired).
# :::

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.3 "Example LO Problems"
#   (transportation), §6.5 "Example ILO Problems." Spreadsheet material replaced with pulp.
# - Dantzig, G.B. (1954). "A comment on Edie's 'Traffic delays at toll booths.'" *Journal of
#   the Operations Research Society of America*, 2(3):339–341.
# - Guenin, B., Könemann, J., & Tunçel, L. (2014). *A Gentle Introduction to Optimization*.
#   Cambridge University Press.
