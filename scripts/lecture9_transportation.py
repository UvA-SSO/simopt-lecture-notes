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
# # Lecture 9: Transportation and Transshipment
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture9_transportation.ipynb)

# %% [markdown]
# This notebook covers the transportation problem and its generalization, the
# transshipment problem: how to ship goods between sources and destinations, or through
# intermediate nodes, at minimum cost. Both are LO problems, needing no integrality, and
# the same shape reappears in many assignment problems.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - recognize and formulate the transportation problem;
# - extend a transportation model to transshipment with flow-conservation constraints.

# %%
import pulp

# %% [markdown]
# ## The Transportation Problem
#
# We must ship a single good from $n$ sources (supply $a_i$ at source $i$) to $m$
# destinations (demand $b_j$ at destination $j$), at a cost $c_{ij}$ per unit shipped on
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
# very large $c_{ij}$. Many assignment problems, such as staff to tasks or students to
# rooms, have this same shape.
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
ship = {
    (i, j): pulp.LpVariable(name=f"x_{i}_{j}", lowBound=0) for (i, j) in cost
}

transport += pulp.lpSum(cost[i, j] * ship[i, j] for (i, j) in cost)
for i, cap in supply.items():
    transport += pulp.lpSum(ship[i, j] for j in demand) <= cap, f"supply_{i}"
for j, req in demand.items():
    transport += pulp.lpSum(ship[i, j] for i in supply) >= req, f"demand_{j}"

transport.solve(pulp.PULP_CBC_CMD(msg=False))
print("shipments:", {k: v.value() for k, v in ship.items() if v.value() > 0})
print("total cost:", transport.objective.value())

# %% [markdown]
# (transshipment-problem)=
# ### Transshipment
#
# If goods can pass through intermediate nodes on the way from sources to destinations, we
# have the *transshipment problem*. It is solved by adding, for every intermediate node $k$,
# a flow-conservation constraint: what comes in must go out:
#
# $$
# \sum_{i} x_{ik} = \sum_{j} x_{kj}.
# $$
#
# This extends to a full network by stacking several layers of intermediate nodes, and it
# is the model behind the shortest-path and maximum-flow problems in
# [Algorithms and Heuristics](lecture11_algorithms-heuristics.ipynb).

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.3 "Example LO Problems"
#   (transportation). Spreadsheet material replaced with pulp.
