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
# # Lecture 8: Integer Optimization
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture8_integer-optimization.ipynb)

# %% [markdown]
# An integer linear optimization (ILO) problem is an LO problem with the extra requirement
# that some or all decision variables take integer values, $x_i \in \{0, 1, 2, \dots\}$, or
# are binary, $x_i \in \{0, 1\}$. (A binary variable is just an integer one with the added
# constraint $x_i \le 1$.)
#
# In pulp this is a one-word change: set a variable's `cat` to `"Integer"` or `"Binary"`
# instead of the default `"Continuous"`. Everything else about building and solving the
# model is the same. Solving it, however, is a different matter: in general ILO is much
# harder than LO, as this notebook and [Complexity](lecture11_complexity.ipynb) explain.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - recognize when a problem needs integer or binary decision variables, and formulate and
#   solve it in pulp;
# - explain why ILO is generally harder to solve than LO;
# - explain how branch and bound finds the optimum of an ILO problem;
# - write any LO problem in the general matrix form;
# - explain why linearity (but not integrality) is essential for efficient solvability, and
#   how integer variables let you model many other nonlinearities without giving that up.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pulp

# %% [markdown]
# ## Why Integer Problems Are Harder
#
# Take the bookcase/desk product-mix problem from
# [Linear Optimization](lecture8_linear-optimization.ipynb) and require whole units of $x$
# and $y$. Its LO optimum was $(x, y) = (3.6, 2.8)$ with profit 24.8, not integer. Two
# things go wrong compared to LO:
#
# - **the optimal corner is no longer feasible**, so the simplex reasoning ("the optimum is
#   at a corner") does not directly help;
# - **rounding the LO optimum is not enough**: rounding both up to $(4, 3)$ violates the
#   oak-panel constraint ($4 + 9 = 13 > 12$), and rounding both down to $(3, 2)$ is feasible
#   but only reaches a profit of 19, well below the true integer optimum. Evaluating nearby
#   integer points tells us little, because finding the true optimum in general needs a
#   systematic search.
#
# Let pulp solve the integer version:

# %%
profit = {"bookcase": 3, "desk": 5}
resource_use = {  # resource_use[resource][product]: units of resource per unit of product
    "oak panels": {"bookcase": 1, "desk": 3},
    "assembly hours": {"bookcase": 2, "desk": 1},
}
available = {"oak panels": 12, "assembly hours": 10}
products = list(profit)

int_mix = pulp.LpProblem(name="integer_product_mix", sense=pulp.LpMaximize)
dec_vars = {
    product: pulp.LpVariable(name=product, lowBound=0, cat="Integer") for product in products
}
int_mix += pulp.lpSum(profit[product] * dec_vars[product] for product in products)
for resource, capacity in available.items():
    int_mix += (
        pulp.lpSum(resource_use[resource][product] * dec_vars[product] for product in products)
        <= capacity
    )
int_mix.solve(pulp.PULP_CBC_CMD(msg=False))
print("integer optimum:", {product: dec_vars[product].value() for product in products})
print("optimal profit:", int_mix.objective.value())

# %% [markdown]
# The picture below shows why rounding is unreliable: the LP-relaxation optimum (the star)
# does not sit on the integer grid, and the nearest lattice points are not necessarily
# feasible or optimal. The ILO optimum (the black dot) is the best *feasible* grid point,
# which can be several steps away from the naive rounding of the relaxation.

# %% tags=["hide-input"]
x_grid = np.linspace(0, 6, 200)
plt.figure(figsize=(5, 5))
plt.plot(x_grid, (12 - x_grid) / 3, color="C0", label=r"$x + 3y \leq 12$ (oak panels)")
plt.plot(x_grid, 10 - 2 * x_grid, color="C1", label=r"$2x + y \leq 10$ (assembly hours)")
y_upper = np.minimum((12 - x_grid) / 3, 10 - 2 * x_grid)
plt.fill_between(x_grid, 0, y_upper, where=(y_upper >= 0), alpha=0.15, label="feasible region")

xs, ys = np.meshgrid(range(7), range(5))
feasible = (xs + 3 * ys <= 12) & (2 * xs + ys <= 10)
plt.scatter(xs[feasible], ys[feasible], color="C2", zorder=3, label="integer feasible points")
plt.scatter(
    xs[~feasible], ys[~feasible], color="lightgrey", zorder=2, label="integer infeasible points"
)

plt.plot(3.6, 2.8, "C3*", markersize=14, zorder=4, label="LP relaxation optimum")
x_int, y_int = dec_vars["bookcase"].value(), dec_vars["desk"].value()
plt.plot(x_int, y_int, "ko", markersize=8, zorder=4, label="ILO optimum")

plt.xlim(0, 6)
plt.ylim(0, 4.5)
plt.xlabel("bookcases $x$")
plt.ylabel("desks $y$")
plt.legend(loc="upper right", fontsize=7)
plt.title("Feasible region with integer lattice points")
plt.show()

# %% [markdown]
# :::{exercise}
# :label: ex-6-9
#
# Solve the larger LO problem from
# [Linear Optimization](lecture8_linear-optimization.ipynb#larger-lo-example) again, once
# requiring all variables integer and once requiring them binary. Compare the optimal
# objective values with the continuous one.
# :::

# %% [markdown]
# ## Branch and Bound
#
# The method used to solve ILO problems exactly is branch and bound. It rests on two ideas,
# stated here for a maximization problem:
#
# 1. The LO relaxation, the same problem with the integer constraints dropped
#    ($x_i \in \{0,1\}$ becomes $0 \le x_i \le 1$; $x_i \in \{0,1,2,\dots\}$ becomes
#    $x_i \ge 0$), is less restrictive, so its optimal value is an *upper bound (UB)* on the
#    ILO optimum.
# 2. Any feasible *integer* solution gives a *lower bound (LB)* on the ILO optimum.
#
# If a subproblem's UB is $\le$ the best LB found so far, that subproblem cannot contain a
# better solution and is eliminated: this is what makes the method cleverer than checking
# every integer point. When a relaxation is non-integer, we branch: pick a fractional
# variable, say $x_j = 2.5$, and create two subproblems, one with $x_j \le 2$ and one with
# $x_j \ge 3$. When a relaxation is already integer, it is a candidate LB and we stop
# branching that subproblem.
#
# For the integer product-mix problem:
#
# - **Root.** LO relaxation optimum $(3.6, 2.8)$, value $24.8$, so UB $= 24.8$. Branch on
#   the fractional $y = 2.8$.
# - **Branch $y \le 2$.** Relaxation optimum $(4, 2)$, value $22$, integer, so LB $= 22$.
# - **Branch $y \ge 3$.** Relaxation optimum $(3, 3)$, value $24$, integer, so LB $= 24$.
#
# The best LB is $24$ from the right branch; the left branch's value $22$ is below it, so it
# is eliminated. Every subproblem is now resolved, and $(3, 3)$ with profit $24$ is the
# proven ILO optimum, matching what pulp reported above. Only three linear relaxations had
# to be solved.
#
# Many LO solvers handle integer constraints this way; the best (proprietary) ones for
# large instances are CPLEX and Gurobi.

# %% [markdown]
# ## The Knapsack Problem
#
# The archetypal binary ILO problem is the knapsack problem: from a set of items, each with
# a *reward* and a *weight*, choose a subset of maximum total reward whose total weight fits
# a capacity. Applications include which items to load in a truck, cutting stock in a steel
# plant, and simple forms of portfolio selection.
#
# Consider capacity 10 and six items:
#
# | | 1 | 2 | 3 | 4 | 5 | 6 |
# |---|---|---|---|---|---|---|
# | reward | 10 | 13 | 18 | 31 | 7 | 15 |
# | weight | 2 | 3 | 4 | 7 | 1 | 3 |
#
# With binary $x_i$ (1 = take item $i$):
#
# $$
# \begin{aligned}
# \text{maximize} \quad & \sum_i r_i x_i \\
# \text{subject to} \quad & \sum_i w_i x_i \le 10 \\
# & x_i \in \{0, 1\} \text{ for all } i.
# \end{aligned}
# $$

# %%
reward = [10, 13, 18, 31, 7, 15]
weight = [2, 3, 4, 7, 1, 3]
capacity = 10
n_items = len(reward)

knapsack = pulp.LpProblem(name="knapsack", sense=pulp.LpMaximize)
take = [pulp.LpVariable(name=f"x_{i + 1}", cat="Binary") for i in range(n_items)]
knapsack += pulp.lpSum(reward[i] * take[i] for i in range(n_items))
knapsack += pulp.lpSum(weight[i] * take[i] for i in range(n_items)) <= capacity
knapsack.solve(pulp.PULP_CBC_CMD(msg=False))
print("take items:", [i + 1 for i in range(n_items) if take[i].value() == 1])
print("total reward:", knapsack.objective.value())

# %% [markdown]
# The `build_knapsack` pattern and this data reappear in
# [Modeling Tools and Solvers](lecture10_modeling-tools.ipynb) when we look at warm starting.
#
# :::{exercise}
# :label: ex-6-10
#
# Take the knapsack solution pulp found above. Verify by hand that no single swap (adding
# one currently-excluded item and removing whatever is needed to stay within capacity)
# improves the total reward.
# :::
#
# :::{exercise}
# :label: ex-6-11
#
# Solve, by branch and bound *on paper*, the knapsack problem with rewards $(15, 9, 10, 5)$,
# weights $(1, 3, 5, 4)$ and capacity 8. For the relaxation bound, fill the capacity with
# items in decreasing order of reward-to-weight ratio, allowing a fraction of the last one.
# Check your answer with pulp.
# :::

# %% [markdown]
# (general-formulation)=
# ## General Formulation
#
# Having now seen both LO and ILO in action, we can step back and write down the general
# form both fit into. An LO problem with $n$ decision variables and $m$ constraints is
#
# $$
# \begin{aligned}
# \text{maximize} \quad & \sum_{j=1}^{n} p_j x_j \\
# \text{subject to} \quad & \sum_{j=1}^{n} a_{ij} x_j \le b_i, \quad i = 1, \dots, m \\
# & x_1, \dots, x_n \ge 0,
# \end{aligned}
# $$
#
# or in matrix notation $\max\{p^T x \mid A x \le b,\ x \ge 0\}$, with $p, x$ column vectors
# of length $n$, $b$ a column vector of length $m$, and $A$ an $m \times n$ matrix.
#
# This one form covers more than it seems. Every other case can be rewritten into it:
#
# - **minimization**: $\min p^T x = -\max (-p^T x)$;
# - **"$\ge$" constraints**: $Ax \ge b \Leftrightarrow -Ax \le -b$;
# - **"$=$" constraints**: $Ax = b \Leftrightarrow Ax \le b$ and $Ax \ge b$;
# - **free (unrestricted) variables**: replace $x$ by $x^+ - x^-$ with $x^+, x^- \ge 0$.

# %% [markdown]
# (why-linearity-matters)=
# ## Why Linearity Matters, and What Integrality Buys Back
#
# Linearity is what makes LO efficiently solvable: written in the
# [general form above](#general-formulation), the feasible region is a convex polyhedron,
# the optimum sits at a corner, and a local optimum is automatically global. As soon as the
# objective or a constraint is nonlinear, both of those break:
#
# - **Nonlinear objective.** Maximize $x_1 x_2$ subject to $x_1 + x_2 \le 1$,
#   $x_1, x_2 \ge 0$. The optimum is $(0.5, 0.5)$, in the *interior* of an edge, not at a
#   corner.
# - **Nonlinear constraint.** Maximize $x_1 + x_2$ subject to $\min(x_1, x_2) = 0$,
#   $x_1 \le 2$, $x_2 \le 1$. The feasible set is two line segments meeting at the origin
#   (either $x_1 = 0$ or $x_2 = 0$). It has two *local* optima, $(2, 0)$ and $(0, 1)$; you
#   cannot be sure which is global without checking both. More generally, the simplex
#   reasoning from [Linear Optimization](lecture8_linear-optimization.ipynb) breaks down
#   whenever the feasible region is not a convex polyhedron: you are not sure you have
#   found the best solution until you have checked every local optimum, which is usually
#   intractable.
#
# Nonlinear optimization therefore needs slower, less reliable algorithms. But there is a
# large and useful middle ground: integer constraints. On the one hand, requiring
# $x_i \in \{0, 1, 2, \dots\}$ is itself a nonlinear constraint, and it does make a problem
# harder to solve, as branch and bound's extra work above shows. On the other hand, unlike
# general nonlinearities, it is not *intractably* so, and many other nonlinearities (an
# either/or choice, a fixed cost that applies only when an activity is used, a "this
# constraint holds only if..." condition) can be expressed with integer (usually binary)
# variables and otherwise-linear constraints, and then solved with branch and bound. That is
# why so much modeling effort goes into casting a problem as ILO, and why it is worth
# treating as its own class rather than lumping it in with general nonlinear optimization.
# [Transportation and Transshipment](lecture9_transportation.ipynb),
# [Set Covering and Shift Scheduling](lecture9_covering.ipynb), and
# [Machine Scheduling](lecture9_machine-scheduling.ipynb) show many of those tricks in
# practice.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.4 "Integer Problems."
