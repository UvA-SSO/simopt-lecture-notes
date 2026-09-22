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
# # Lecture 10: Modeling Tools and Solvers

# %% [markdown]
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture10_modeling-tools.ipynb)

# %% [markdown]
# This notebook has two halves. First, two more applications that need a modeling trick:
# multi-period inventory planning and robust regression. Then the tooling: the solvers that
# actually do the optimizing, and the modeling tools (algebraic modeling languages, and
# pulp) that sit between your problem and a solver.
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

try:
    import pulp
except ModuleNotFoundError:  # pulp is not preinstalled on Google Colab
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pulp"], check=True)
    import pulp

# %% [markdown]
# (production-inventory-model)=
# ## Multi-Period Inventory Planning
#
# A system whose state is tracked over time is a multi-period model. The classic case
# is inventory: we hold a single product, start with stock $s_0$, and for each period
# $t = 1, \dots, T$ we know the demand $d_t$, the holding cost $h_t$ per unit left at the
# end of the period, and the order cost $c_t$ per unit ordered. The decision is how much to
# order each period, $x_t$; the resulting end-of-period stock is $s_t$. Stock evolves as
#
# $$
# s_t = s_{t-1} + x_t - d_t,
# $$
#
# and requiring $s_t \ge 0$ forbids backorders (all demand must be met). The LO model:
#
# $$
# \begin{aligned}
# \text{minimize} \quad & \sum_{t=1}^{T} (c_t x_t + h_t s_t) \\
# \text{subject to} \quad & s_t = s_{t-1} + x_t - d_t \text{ for } t = 1, \dots, T \\
# & x_t, s_t \ge 0 \text{ for } t = 1, \dots, T.
# \end{aligned}
# $$
#
# The parameters $d_t, h_t, c_t$ typically come from a forecast. The model extends easily to
# a maximum stock level, a production capacity, several products, or fixed order costs.

# %%
demand = [7, 9, 5, 8]
holding = [1, 1, 1, 1]
order_cost = [8, 11, 7, 10]
s0 = 6
periods = range(len(demand))

inventory = pulp.LpProblem(name="inventory", sense=pulp.LpMinimize)
order = [pulp.LpVariable(name=f"x_{t + 1}", lowBound=0) for t in periods]
stock = [pulp.LpVariable(name=f"s_{t + 1}", lowBound=0) for t in periods]

inventory += pulp.lpSum(order_cost[t] * order[t] + holding[t] * stock[t] for t in periods)
for t in periods:
    prev = s0 if t == 0 else stock[t - 1]
    inventory += stock[t] == prev + order[t] - demand[t], f"balance_{t + 1}"

inventory.solve(pulp.PULP_CBC_CMD(msg=False))
print("orders:", [order[t].value() for t in periods])
print("end-of-period stock:", [stock[t].value() for t in periods])
print("total cost:", inventory.objective.value())

# %% [markdown]
# :::{exercise}
# :label: ex-6-19
#
# Extend the multi-period model with fixed order costs: a cost $K$ is incurred in period
# $t$ whenever $x_t > 0$, regardless of the amount. Keep all constraints linear (hint: a
# binary "did we order in period $t$" variable and a big $M$, as in
# [Machine Scheduling](lecture9_advanced-modeling.ipynb)). Solve with pulp.
# :::

# %% [markdown]
# ## Robust Regression
#
# Ordinary least-squares regression minimizes the sum of *squared* errors and is sensitive
# to outliers, just as the mean is. Minimizing the sum of *absolute* errors instead gives a
# robust fit, the line analogue of the median. Given points $(x_i, y_i)$, we want $a, b$
# solving
#
# $$
# \min \sum_i |y_i - (a + b x_i)|.
# $$
#
# The absolute value is nonlinear, but there is a standard trick: write each error as a
# difference of two non-negative parts, $e_i = e_i^+ - e_i^-$, and put $e_i^+ + e_i^-$ in
# the objective. Because we minimize their sum, at the optimum one of the two is always 0,
# so $e_i^+ + e_i^- = |e_i|$:
#
# $$
# \min \sum_i (e_i^+ + e_i^-) \quad\text{s.t.}\quad y_i - (a + b x_i) = e_i^+ - e_i^-,\quad
# e_i^+, e_i^- \ge 0.
# $$
#
# The same $x = x^+ - x^-$ split linearizes any $|x|$ that appears (with a non-negative
# coefficient) in an objective. Weighting the two parts differently,
# $p \sum e_i^+ + (1 - p) \sum e_i^-$ with $0 < p < 1$, tilts the line toward the upper or
# lower points: this is quantile regression ([](#fig-quantile-regression)).

# %% [markdown]
# :::{figure} images/lecture9_fig6.11.png
# :label: fig-quantile-regression
#
# Quantile regression: the fitted line for different quantile levels $p$.
# :::

# %%
xs = np.array([2, 4, 6, 8, 10])
ys = np.array([5, 4, 9, 3, 7])

fit = pulp.LpProblem(name="robust_regression", sense=pulp.LpMinimize)
a = pulp.LpVariable(name="a")  # free (default bounds -inf..inf)
slope = pulp.LpVariable(name="b")
e_pos = [pulp.LpVariable(name=f"ep_{k}", lowBound=0) for k in range(len(xs))]
e_neg = [pulp.LpVariable(name=f"en_{k}", lowBound=0) for k in range(len(xs))]

fit += pulp.lpSum(e_pos[k] + e_neg[k] for k in range(len(xs)))
for k in range(len(xs)):
    fit += ys[k] - (a + slope * xs[k]) == e_pos[k] - e_neg[k]

fit.solve(pulp.PULP_CBC_CMD(msg=False))
a_hat, b_hat = a.value(), slope.value()
print(f"robust line: y = {a_hat:.2f} + {b_hat:.2f} x")

plt.figure(figsize=(5, 3.5))
plt.scatter(xs, ys)
grid = np.linspace(xs.min(), xs.max(), 50)
plt.plot(grid, a_hat + b_hat * grid, color="grey")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Least-absolute-deviation fit")
plt.show()

# %% [markdown]
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
# Take [the call-center staffing exercise](lecture9_ilo-applications.ipynb#ex-6-13) with
# only 8-hour shifts. Instead of requiring the staffing level to be met in every interval,
# minimize the sum of absolute differences between staffing and demand. Formulate as an LO
# and solve with pulp.
# :::
#
# The trick also works when the absolute value is a *penalty* rather than the whole
# objective. For the product-mix problem, suppose we dislike making the two products in very
# different quantities and add $-|b - d|$ to the profit. Introduce $\delta^+, \delta^- \ge 0$
# with $b - d = \delta^+ - \delta^-$ and subtract $\delta^+ + \delta^-$ from the objective.

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
# (`pip install pulp[cbc]` bundles CBC), plus everything Python brings for preparing data
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
    name="after", indices=[(p, q) for p in jobs for q in jobs if p != q], cat="Binary"
)
print(start["B"], "/", after[("A", "B")])

# %% [markdown]
# Adding an expression *without* a comparison registers the objective; *with* a comparison,
# a constraint. The optional trailing string names it, helpful when you `print` the
# problem or read the solver log.

# %%
demo = pulp.LpProblem(name="demo", sense=pulp.LpMinimize)
demo += pulp.lpSum(start[job] + duration[job] for job in jobs), "sum_of_finish_times"
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


def build_knapsack(capacity: float) -> tuple[pulp.LpProblem, list[pulp.LpVariable]]:
    problem = pulp.LpProblem(name="knapsack", sense=pulp.LpMaximize)
    items = [pulp.LpVariable(name=f"x_{i + 1}", cat="Binary") for i in range(len(reward))]
    problem += pulp.lpSum(reward[i] * items[i] for i in range(len(reward)))
    problem += pulp.lpSum(weight[i] * items[i] for i in range(len(reward))) <= capacity
    return problem, items


knap10, items10 = build_knapsack(capacity=10)
knap10.solve(pulp.PULP_CBC_CMD(msg=False))
print("capacity 10:", [v.value() for v in items10], knap10.objective.value())

knap11, items11 = build_knapsack(capacity=11)
for new, old in zip(items11, items10):
    new.setInitialValue(old.value())
knap11.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=True))
print("capacity 11, warm started:", [v.value() for v in items11], knap11.objective.value())

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
# Warm start [the shift-scheduling exercise](lecture9_ilo-applications.ipynb#ex-6-13) from
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
