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
# description: "Multi-period inventory planning and robust regression as optimization models, and the solvers and modeling tools such as pulp that solve them."
# thumbnail: null
# ---
# # Lecture 10: Modeling Tools and Solvers
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture10_modeling-tools.ipynb)

# %% [markdown]
# This notebook has two halves. First, two more applications that need a modeling trick:
# multi-period inventory planning and robust regression. They are introduced in the same
# steps as the applications of [Lecture 9](lecture9_introduction.ipynb): practical
# motivation, generic model, the model for an example, its solution in pulp, and
# extensions. Then the tooling: the solvers that actually do the optimizing, and the
# modeling tools (algebraic modeling languages, and pulp) that sit between your problem
# and a solver.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - model multi-period problems with state variables, such as inventory planning;
# - formulate a robust regression problem as an LO model;
# - describe the roles of solvers versus modeling tools, and choose between them.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pulp

# %% [markdown]
# (production-inventory-model)=
# ## Multi-Period Inventory Planning
#
# ### Practical Motivation
#
# A shop sells a product whose demand varies from week to week, and the purchase price
# varies as well (because of supplier promotions, or seasonal prices). Ordering a lot
# when the price is low saves money, but products that sit in stock cost money too:
# storage space, capital, insurance. When should the shop order, and how much, to meet
# all demand at the lowest total cost? This is an example of a **multi-period model**: a
# model in which the state of a system (here, the stock) is tracked over time, and a
# decision in one period affects all later periods.

# %% [markdown]
# ### Modeling
#
# #### Problem Definition and Example
#
# We hold a single product and plan for periods $t = 1, \dots, T$ (the time horizon). At
# the start we have $s_0$ units in stock. For every period $t$ we know the demand $d_t$,
# the order cost $c_t$ per unit ordered in period $t$, and the holding cost $h_t$ per unit
# left in stock at the end of period $t$. An order arrives immediately. The problem is to
# decide how much to order in each period, such that all demand is met from stock (no
# backorders) at minimal total order and holding cost. The parameters $d_t$, $c_t$ and
# $h_t$ typically come from a forecast (predictive analytics).
#
# As an example, take $T = 4$ periods with an initial stock of $s_0 = 6$ and
#
# | period $t$ | 1 | 2 | 3 | 4 |
# |---|---|---|---|---|
# | demand $d_t$ | 7 | 9 | 5 | 8 |
# | order cost $c_t$ | 8 | 11 | 7 | 10 |
# | holding cost $h_t$ | 1 | 1 | 1 | 1 |
#
# #### Decision Variables
#
# The decisions are the order quantities
#
# $$
# x_t = \text{number of units ordered in period } t, \quad t = 1, \dots, T.
# $$
#
# We also introduce the stock at the end of each period as a decision variable,
#
# $$
# s_t = \text{number of units in stock at the end of period } t, \quad t = 1, \dots, T.
# $$
#
# Strictly speaking, $s_t$ is not a free choice: once the orders are fixed, the stock
# follows from them. We could substitute $s_t = s_0 + \sum_{k \le t} (x_k - d_k)$
# everywhere, but that makes every constraint and the objective much longer. A variable
# like $s_t$, which describes the state of the system in a period, is called a **state
# variable**; it keeps a multi-period model short and readable.
#
# #### Objective
#
# In period $t$ we pay $c_t x_t$ for ordering and $h_t s_t$ for holding stock, so the
# total cost is
#
# $$
# \sum_{t=1}^{T} (c_t x_t + h_t s_t).
# $$
#
# #### Constraints
#
# The stock at the end of period $t$ is the stock at the end of the previous period,
# plus what is ordered, minus what is sold:
#
# $$
# s_t = s_{t-1} + x_t - d_t, \quad t = 1, \dots, T.
# $$
#
# This **balance constraint** links each period to the previous one; for $t = 1$,
# $s_0$ is the given initial stock. Orders cannot be negative, $x_t \ge 0$. The
# requirement $s_t \ge 0$ is what forbids backorders: if demand in period $t$ were not
# met from stock, the balance constraint would give a negative $s_t$.
#
# #### Complete LO Model
#
# $$
# \begin{aligned}
# \min \quad & \sum_{t=1}^{T} (c_t x_t + h_t s_t) \\
# \text{s.t.} \quad & s_t = s_{t-1} + x_t - d_t, \quad t = 1, \dots, T \\
# & x_t, s_t \ge 0, \quad t = 1, \dots, T.
# \end{aligned}
# $$

# %% [markdown]
# ### Modeling the Example
#
# $$
# \begin{aligned}
# \min \quad & 8x_1 + 11x_2 + 7x_3 + 10x_4 + s_1 + s_2 + s_3 + s_4 \\
# \text{s.t.} \quad & s_1 = 6 + x_1 - 7 \\
# & s_2 = s_1 + x_2 - 9 \\
# & s_3 = s_2 + x_3 - 5 \\
# & s_4 = s_3 + x_4 - 8 \\
# & x_1, x_2, x_3, x_4, s_1, s_2, s_3, s_4 \ge 0.
# \end{aligned}
# $$

# %% [markdown]
# ### Solving the Example in pulp
#
# The balance constraints are equalities, written with `==` in pulp. Python lists start
# at index 0, so period $t$ is `t + 1` in the variable names, and the first period uses
# the initial stock `s0` in place of the previous period's stock variable.

# %%
demand = [7, 9, 5, 8]
holding = [1, 1, 1, 1]
order_cost = [8, 11, 7, 10]
s0 = 6
periods = range(len(demand))

inventory = pulp.LpProblem(name="inventory", sense=pulp.LpMinimize)
order = [pulp.LpVariable(name=f"x_{t + 1}", lowBound=0) for t in periods]
stock = [pulp.LpVariable(name=f"s_{t + 1}", lowBound=0) for t in periods]

inventory += pulp.lpSum(
    order_cost[t] * order[t] + holding[t] * stock[t] for t in periods
)
for t in periods:
    prev = s0 if t == 0 else stock[t - 1]
    inventory += stock[t] == prev + order[t] - demand[t], f"balance_{t + 1}"

inventory.solve(pulp.PULP_CBC_CMD(msg=False))
print("status:", pulp.LpStatus[inventory.status])
print("orders:", [order[t].value() for t in periods])
print("end-of-period stock:", [stock[t].value() for t in periods])
print("total cost:", inventory.objective.value())

# %% [markdown]
# The optimal plan orders only in the two cheap periods, 1 and 3. Ordering period 2's
# demand in period 1 costs $8 + 1 = 9$ per unit (order plus one period of holding),
# which is less than the 11 of ordering it in period 2; period 4's demand is ordered in
# period 3 for the same reason.

# %% [markdown]
# ### Extensions
#
# The model extends easily; each of the following adds a constraint or an index, but no
# new modeling trick:
#
# - **maximum stock level**: $s_t \le S$ for a storage capacity $S$;
# - **production capacity**: $x_t \le P_t$ if the product is made rather than bought;
# - **several products**: give every variable and parameter a product index, and add
#   constraints for resources the products share, such as storage space.
#
# Fixed order costs, which are paid in every period in which something is ordered, do
# need a new trick:
#
# :::{exercise}
# :label: ex-6-19
#
# Extend the multi-period model with fixed order costs: a cost $K$ is incurred in period
# $t$ whenever $x_t > 0$, regardless of the amount. Keep all constraints linear (hint: a
# binary "did we order in period $t$" variable and a big $M$, as in
# [Machine Scheduling](lecture9_machine-scheduling.ipynb)). Solve with pulp.
# :::

# %% [markdown]
# ## Robust Regression
#
# ### Practical Motivation
#
# A factory wants to predict how long a production order will take from its size, to
# promise delivery dates. It fits a line through the data of past orders. A few of those
# orders took far longer than usual, for example because a machine broke down.
# Ordinary least-squares (OLS) regression minimizes the sum of *squared* prediction
# errors, so such outliers have a large influence and pull the line toward them, just as
# a single extreme value pulls the mean. Minimizing the sum of *absolute* errors instead
# gives a line that is far less sensitive to outliers, the line analogue of the median.
# This is called **robust regression** (or least absolute deviation regression).

# %% [markdown]
# ### Modeling
#
# #### Problem Definition and Example
#
# Given $n$ data points $(x_i, y_i)$, $i = 1, \dots, n$, find the line $a + bx$ that
# minimizes the sum of the absolute prediction errors $|y_i - (a + b x_i)|$.
#
# As an example, take the five data points
#
# | $i$ | 1 | 2 | 3 | 4 | 5 |
# |---|---|---|---|---|---|
# | $x_i$ | 2 | 4 | 6 | 8 | 10 |
# | $y_i$ | 5 | 4 | 9 | 3 | 7 |
#
# #### Decision Variables
#
# We choose the line, so its intercept $a$ and slope $b$ are decision variables. Both
# may be negative, so unlike in the previous models they have no sign constraint. It
# helps to also introduce the prediction error of each data point as a variable,
#
# $$
# e_i = y_i - (a + b x_i), \quad i = 1, \dots, n.
# $$
#
# #### Objective
#
# We want to minimize $\sum_{i=1}^{n} |e_i|$. The absolute value is not linear, so this
# is not yet an LO model. There is a standard trick to fix this. Write each error as the
# difference of two non-negative variables,
#
# $$
# e_i = e_i^+ - e_i^-, \quad e_i^+, e_i^- \ge 0,
# $$
#
# and replace $|e_i|$ by $e_i^+ + e_i^-$ in the objective, which is linear:
#
# $$
# \min \sum_{i=1}^{n} (e_i^+ + e_i^-).
# $$
#
# Why is this allowed? The same $e_i$ can be written as $e_i^+ - e_i^-$ in many ways,
# for example $2 = 2 - 0 = 5 - 3$. But if both $e_i^+$ and $e_i^-$ were positive, we
# could lower both by the same amount: $e_i$ stays the same and the objective goes down.
# So at the optimum, at least one of the two is 0, and then $e_i^+ + e_i^- = |e_i|$. If
# $e_i^+ > 0$, the data point lies $e_i^+$ above the line; if $e_i^- > 0$, it lies
# $e_i^-$ below the line.
#
# #### Constraints
#
# The only constraints connect the errors to the line:
#
# $$
# e_i^+ - e_i^- = y_i - (a + b x_i), \quad i = 1, \dots, n,
# $$
#
# together with $e_i^+, e_i^- \ge 0$. The variables $a$ and $b$ are free.
#
# #### Complete LO Model
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i=1}^{n} (e_i^+ + e_i^-) \\
# \text{s.t.} \quad & e_i^+ - e_i^- = y_i - (a + b x_i), \quad i = 1, \dots, n \\
# & e_i^+, e_i^- \ge 0, \quad i = 1, \dots, n.
# \end{aligned}
# $$

# %% [markdown]
# ### Modeling the Example
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i=1}^{5} (e_i^+ + e_i^-) \\
# \text{s.t.} \quad & e_1^+ - e_1^- = 5 - (a + 2b) \\
# & e_2^+ - e_2^- = 4 - (a + 4b) \\
# & e_3^+ - e_3^- = 9 - (a + 6b) \\
# & e_4^+ - e_4^- = 3 - (a + 8b) \\
# & e_5^+ - e_5^- = 7 - (a + 10b) \\
# & e_i^+, e_i^- \ge 0, \quad i = 1, \dots, 5.
# \end{aligned}
# $$

# %% [markdown]
# ### Solving the Example in pulp
#
# A variable without `lowBound` is free in pulp (its bounds default to $-\infty$ and
# $\infty$), which is what we need for the intercept and the slope. The data points are
# stored as numpy arrays, which pulp's expressions accept like ordinary numbers.

# %%
xs = np.array([2, 4, 6, 8, 10])
ys = np.array([5, 4, 9, 3, 7])
points = range(len(xs))

robust_regression = pulp.LpProblem(
    name="robust_regression", sense=pulp.LpMinimize
)
intercept = pulp.LpVariable(name="a")
slope = pulp.LpVariable(name="b")
e_pos = [pulp.LpVariable(name=f"ep_{k + 1}", lowBound=0) for k in points]
e_neg = [pulp.LpVariable(name=f"en_{k + 1}", lowBound=0) for k in points]

robust_regression += pulp.lpSum(e_pos[k] + e_neg[k] for k in points)
for k in points:
    prediction = intercept + slope * xs[k]
    robust_regression += ys[k] - prediction == e_pos[k] - e_neg[k]

robust_regression.solve(pulp.PULP_CBC_CMD(msg=False))
a_hat, b_hat = intercept.value(), slope.value()
print("status:", pulp.LpStatus[robust_regression.status])
print(f"line: y = {a_hat:.2f} + {b_hat:.2f} x")
print("sum of absolute errors:", robust_regression.objective.value())

# %% tags=["remove-cell"] label="robust-fit"
plt.figure(figsize=(5, 3.5))
plt.scatter(xs, ys)
grid = np.linspace(xs.min(), xs.max(), 50)
plt.plot(grid, a_hat + b_hat * grid, color="grey")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %% [markdown]
# :::{figure} #robust-fit
# :label: fig-robust-fit
#
# The five data points of the example and the line $y = 4.5 + 0.25x$ that minimizes the
# sum of absolute errors.
# :::
#
# The optimal line passes exactly through the first and the last data point. That is no
# coincidence: as in every LO problem, there is an optimal solution in a corner point,
# and for this model that means a line through (at least) two of the data points.

# %% [markdown]
# ### Extensions
#
# #### Quantile Regression
#
# Weighting the two parts of the error differently,
# $\min\, p \sum_i e_i^+ + (1 - p) \sum_i e_i^-$ with $0 < p < 1$, tilts the line toward
# the upper or lower points: this is quantile regression ([](#fig-quantile-regression)).
# With $p = 0.5$ it is the robust regression above; with $p = 0.9$, about 90% of the data
# points lie below the line, which is useful for promising delivery times that are met
# in 90% of the cases.
#
# :::{figure} images/lecture9_fig6.11.png
# :label: fig-quantile-regression
#
# Quantile regression: the fitted line for different quantile levels $p$.
# :::
#
# #### Absolute Values in Other Models
#
# The same $x = x^+ - x^-$ split linearizes any $|x|$ that appears in an objective that
# is minimized with a non-negative coefficient (or maximized with a non-positive one).
# The trick also works when the absolute value is a *penalty* rather than the whole
# objective. For the product-mix problem, suppose we dislike making the two products in
# very different quantities and add $-|x - y|$ to the profit. Introduce
# $\delta^+, \delta^- \ge 0$ with $x - y = \delta^+ - \delta^-$ and subtract
# $\delta^+ + \delta^-$ from the objective.
#
# :::{exercise}
# :label: ex-6-16
#
# Numbers $a_1, \dots, a_n$ are given; find $x$ minimizing $\sum_i |x - a_i|$. Formulate as
# an LO and solve in pulp for 1, 2, 3, 5, 8, 10, 20, 35, 100. How do you interpret the
# result?
# :::
#
# :::{exercise}
# :label: ex-6-17
#
# Take [the call-center staffing exercise](lecture9_exercises.ipynb#ex-6-13) with
# only 8-hour shifts. Instead of requiring the staffing level to be met in every interval,
# minimize the sum of absolute differences between staffing and demand. Formulate as an LO
# and solve with pulp.
# :::

# %% [markdown]
# ## Solvers
#
# The solver is the engine that does the optimizing. Very roughly:
#
# | | examples | notes |
# |---|---|---|
# | open-source | CBC (pulp's default), HiGHS, GLPK | free; fine for small and medium problems |
# | commercial | Gurobi, CPLEX, FICO Xpress | fastest on hard/large ILO; free academic licenses |
# | spreadsheet | Excel Solver, OpenSolver | Excel Solver is weak; OpenSolver embeds CBC |
#
# For LO, the classic method is the simplex algorithm (corner to corner). Since the 1980s,
# interior-point methods move through the interior of the feasible region instead and solve
# LO in provably polynomial time; modern solvers offer both. For ILO, branch and bound (see
# [Integer Optimization](lecture8_integer-optimization.ipynb)) wraps around an LO solver.
# ILO solver performance has improved by a factor of roughly 1000 between 2000 and 2020
# through better algorithms alone, comparable to the hardware speed-up over the same
# period.
#
# pulp can call any installed solver without changing the model; only the `.solve(...)`
# line changes. To see what is available here:

# %%
print(pulp.listSolvers(onlyAvailable=True))

# %% [markdown]
# Without installing anything, you can also export the model to a standard `.mps` file and
# submit it to the free [NEOS Server](https://neos-server.org/neos/), which hosts many
# solvers including commercial ones. If the variable and constraint names might leak
# information about your data, pulp can anonymize them on export with `rename=1`.

# %% [markdown]
# ## Modeling Tools
#
# Most of an optimization specialist's time goes into modeling, not solving. An algebraic
# modeling language (AML), such as AMPL, AIMMS, or GAMS, is a language for writing a model
# in near-mathematical notation, kept separate from the data, and handed to whichever
# solver you choose. AMLs are quick to write, easy to communicate, and let you re-solve new
# instances by swapping only the data; the downsides are cost, closed source, and awkward
# embedding in other software. A decision support system (DSS) goes the other way: a solver
# built into software for one specific task (vehicle routing, room pricing), used by domain
# planners rather than modelers.
#
# [pulp](https://coin-or.github.io/pulp/) gives the AML benefits inside Python
# (`pip install pulp` includes the CBC solver), plus everything Python brings for preparing data
# and analyzing solutions. The key discipline is the same as an AML's model/data split:
# keep the instance data in plain Python structures, and build the model (`LpProblem`,
# variables, objective, constraints) from that data so the model code is reused unchanged
# for a new instance.

# %%
jobs = ["A", "B", "C"]
duration = {"A": 6, "B": 4, "C": 5}

# %% [markdown]
# `pulp.LpVariable.dicts` builds a dictionary of variables from a list of keys, which reads
# more naturally than a list once there are several variable groups:

# %%
start = pulp.LpVariable.dicts(name="start", indices=jobs, lowBound=0)
after = pulp.LpVariable.dicts(
    name="after",
    indices=[(p, q) for p in jobs for q in jobs if p != q],
    cat="Binary",
)
print(start["B"], "/", after[("A", "B")])

# %% [markdown]
# Adding an expression *without* a comparison registers the objective; *with* a comparison,
# a constraint. The optional trailing string names it, helpful when you `print` the
# problem or read the solver log.

# %%
demo = pulp.LpProblem(name="demo", sense=pulp.LpMinimize)
demo += (
    pulp.lpSum(start[job] + duration[job] for job in jobs),
    "sum_of_finish_times",
)
for job in jobs:
    demo += start[job] <= 10, f"{job}_starts_by_10"
print(demo)

# %% [markdown]
# `print(problem)` shows every variable, the objective, and every named constraint: the
# first thing to do when a model misbehaves, especially on a small instance. Passing
# `msg=True` to the solver shows its log; for CBC on an ILO the log reports each improved
# integer solution and the remaining optimality gap (the guaranteed distance to the best
# possible value). A run ending `Optimal` with gap 0% has *proven* optimality; a run
# stopped early reports the best solution so far and the still-open gap.

# %%
demo.solve(pulp.PULP_CBC_CMD(msg=True))
print("status:", pulp.LpStatus[demo.status])

# %% [markdown]
# ## Beyond the Lecture: Warm Starting
#
# When you re-solve a problem that is *almost* the same as one already solved (the same
# knapsack with one more item, the same schedule with one duration changed), you can hand
# the solver the old solution to warm start from: `.setInitialValue(...)` on each variable,
# then solve with `warmStart=True`.

# %%
reward = [10, 13, 18, 31, 7, 15]
weight = [2, 3, 4, 7, 1, 3]


def build_knapsack(
    capacity: float,
) -> tuple[pulp.LpProblem, list[pulp.LpVariable]]:
    problem = pulp.LpProblem(name="knapsack", sense=pulp.LpMaximize)
    items = [
        pulp.LpVariable(name=f"x_{i + 1}", cat="Binary")
        for i in range(len(reward))
    ]
    problem += pulp.lpSum(reward[i] * items[i] for i in range(len(reward)))
    problem += (
        pulp.lpSum(weight[i] * items[i] for i in range(len(reward)))
        <= capacity
    )
    return problem, items


knap10, items10 = build_knapsack(capacity=10)
knap10.solve(pulp.PULP_CBC_CMD(msg=False))
print("capacity 10:", [v.value() for v in items10], knap10.objective.value())

knap11, items11 = build_knapsack(capacity=11)
for new, old in zip(items11, items10):
    new.setInitialValue(old.value())
knap11.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=True))
print(
    "capacity 11, warm started:",
    [v.value() for v in items11],
    knap11.objective.value(),
)

# %% [markdown]
# For a problem this small the effect is not measurable, but inside a loop that re-solves a
# large ILO with small data changes (for example the simulation-optimization loop in
# [Simulation Optimization](lecture13_simulation-optimization.ipynb)), a warm start can cut
# the number of branch-and-bound nodes noticeably.
#
# :::{exercise}
# :label: ex-10-1
#
# Re-solve the knapsack from [Integer Optimization](lecture8_integer-optimization.ipynb)
# with `msg=True` and find, in the log, the point where CBC first proves optimality
# (gap 0%).
# :::
#
# :::{exercise}
# :label: ex-10-2
#
# Warm start [the shift-scheduling exercise](lecture9_exercises.ipynb#ex-6-13) from
# the previous day's optimal schedule when one interval's demand changes slightly. Compare
# the number of explored nodes (from the `msg=True` log) with and without the warm start.
# :::

# %% [markdown]
# ## Beyond the Lecture: Debugging with a GenAI Assistant
#
# A GenAI assistant (Claude, ChatGPT, ...) can be a useful pulp pair-programmer if used
# carefully:
#
# - paste the actual code and the actual error or solver log, not a paraphrase;
# - always re-check a suggested model against your original formulation: it can flip a
#   constraint's direction, quietly redefine a variable, or shift an index range, and pulp
#   will build and solve the wrong model without complaint;
# - `print(problem)` before and after any suggested change shows exactly what moved;
# - for an `"Infeasible"` or `"Unbounded"` status, give it the full model text and ask it to
#   find the conflicting or missing constraint.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.3 (multi-period), §6.6
#   "Modeling Tools", §6.7 "Modeling Tricks" (quantile regression). Spreadsheet and AMPL
#   material replaced with pulp.
# - PuLP documentation: https://coin-or.github.io/pulp/
# - `Course Materials/pulp_tutorial.py` (this course's PuLP tutorial, by Joost Berkhout).
