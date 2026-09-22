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

# %% [markdown]
# An **integer linear optimization (ILO)** problem is an LO problem with the extra
# requirement that some or all decision variables take integer values,
# $x_i \in \{0, 1, 2, \dots\}$, or are **binary**, $x_i \in \{0, 1\}$. (A binary variable
# is just an integer one with the added constraint $x_i \le 1$.)
#
# In pulp this is a one-word change: set a variable's `cat` to `"Integer"` or `"Binary"`
# instead of the default `"Continuous"`. Everything else about building and solving the
# model is the same. Solving it, however, is a different matter — in general ILO is much
# harder than LO, as this notebook and [Complexity](lecture11_complexity.ipynb) explain.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - recognize when a problem needs integer or binary decision variables, and formulate and
#   solve it in pulp;
# - explain why ILO is generally harder to solve than LO;
# - explain how branch and bound finds the optimum of an ILO problem.

# %%
import pulp

# %% [markdown]
# ## Why Integer Problems Are Harder
#
# Take the product-mix problem from [Linear Optimization](lecture8_linear-optimization.ipynb)
# and require whole bookcases and desks. Its LO optimum was $(b, d) = (3.6, 2.8)$ with
# profit 24.8 — not integer. Two things go wrong compared to LO:
#
# - **the optimal corner is no longer feasible**, so the simplex reasoning ("the optimum is
#   at a corner") does not directly help;
# - **rounding the LO optimum is not enough**: $(4, 3)$ violates the oak-panel constraint
#   ($4 + 9 = 13 > 12$), and $(4, 2)$ or $(3, 3)$ each need checking. Evaluating the corners
#   tells us little, because the integer optimum sits somewhere *inside* the feasible
#   region.
#
# Let pulp solve the integer version:

# %%
profit = {"bookcase": 3, "desk": 5}
use = {"oak panels": {"bookcase": 1, "desk": 3}, "assembly hours": {"bookcase": 2, "desk": 1}}
available = {"oak panels": 12, "assembly hours": 10}
products = list(profit)

int_mix = pulp.LpProblem(name="integer_product_mix", sense=pulp.LpMaximize)
q = {p: pulp.LpVariable(name=p.replace(" ", "_"), lowBound=0, cat="Integer") for p in products}
int_mix += pulp.lpSum(profit[p] * q[p] for p in products)
for r, cap in available.items():
    int_mix += pulp.lpSum(use[r][p] * q[p] for p in products) <= cap
int_mix.solve(pulp.PULP_CBC_CMD(msg=False))
print("integer optimum:", {p: q[p].value() for p in products}, "profit", int_mix.objective.value())

# %% [markdown]
# :::{exercise}
# :label: ex-6-9
#
# Solve the larger LO problem from [Linear Optimization](lecture8_linear-optimization.ipynb)
# again, once requiring all variables integer and once requiring them binary. Compare the
# optimal objective values with the continuous one.
# :::

# %% [markdown]
# ## Branch and Bound
#
# The method used to solve ILO problems exactly is **branch and bound**. It rests on two
# ideas, stated here for a maximization problem:
#
# 1. The **LO relaxation** — the same problem with the integer constraints dropped
#    ($x_i \in \{0,1\}$ becomes $0 \le x_i \le 1$; $x_i \in \{0,1,2,\dots\}$ becomes
#    $x_i \ge 0$) — is less restrictive, so its optimal value is an **upper bound (UB)** on
#    the ILO optimum.
# 2. Any feasible *integer* solution gives a **lower bound (LB)** on the ILO optimum.
#
# If a subproblem's UB is $\le$ the best LB found so far, that subproblem cannot contain a
# better solution and is **eliminated** — this is what makes the method cleverer than
# checking every integer point. When a relaxation is non-integer, we **branch**: pick a
# fractional variable, say $x_j = 2.5$, and create two subproblems, one with $x_j \le 2$ and
# one with $x_j \ge 3$. When a relaxation is already integer, it is a candidate LB and we
# stop branching that subproblem.
#
# For the integer product-mix problem:
#
# - **Root.** LO relaxation optimum $(3.6, 2.8)$, value $24.8$ → UB $= 24.8$. Branch on the
#   fractional $d = 2.8$.
# - **Branch $d \le 2$.** Relaxation optimum $(4, 2)$, value $22$ — integer, so LB $= 22$.
# - **Branch $d \ge 3$.** Relaxation optimum $(3, 3)$, value $24$ — integer, so LB $= 24$.
#
# The best LB is $24$ from the right branch; the left branch's value $22$ is below it, so it
# is eliminated. Every subproblem is now resolved, and $(3, 3)$ with profit $24$ is the
# proven ILO optimum — matching what pulp reported above. Only three linear relaxations had
# to be solved.
#
# Many LO solvers handle integer constraints this way; the best (proprietary) ones for
# large instances are CPLEX and Gurobi.

# %% [markdown]
# ## The Knapsack Problem
#
# The archetypal binary ILO problem is the **knapsack problem**: from a set of items, each
# with a *reward* and a *weight*, choose a subset of maximum total reward whose total weight
# fits a capacity. Applications include which items to load in a truck, cutting stock in a
# steel plant, and simple forms of portfolio selection.
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
# Take the knapsack solution pulp found above. Verify by hand that no single swap — adding
# one currently-excluded item and removing whatever is needed to stay within capacity —
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
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.4 "Integer Problems."
