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
# description: "The transportation and transshipment problems: shipping goods between sources and destinations, possibly through intermediate nodes, at minimum cost."
# thumbnail: null
# ---
# # Lecture 9: Transportation and Transshipment
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture9_transportation.ipynb)

# %% [markdown]
# This notebook covers the transportation problem and its generalization, the
# transshipment problem: how to ship goods between sources and destinations, or through
# intermediate nodes, at minimum cost. Both are LO problems, needing no integrality, and
# the same shape reappears in many assignment problems. Like every application in this
# lecture, the problem is introduced in the same steps: a practical motivation, the
# generic model built up with the [four modeling
# steps](lecture8_linear-optimization.ipynb#modeling-approach), the model for a concrete
# example, its solution in pulp, and possible extensions.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - recognize and formulate the transportation problem;
# - use decision variables with multiple indices, such as $x_{ij}$, in a model and in pulp;
# - extend a transportation model to transshipment with flow-conservation constraints.

# %% [markdown]
# ## The Transportation Problem
#
# ### Practical Motivation
#
# A company stores its product in a few warehouses and delivers it to many customers.
# Every week, each customer orders a certain amount, each warehouse has a limited stock,
# and shipping one unit from a warehouse to a customer has a cost that depends on the
# distance between them. Which warehouse should deliver how much to which customer, so
# that all orders are delivered at the lowest total transport cost? The same question
# comes up for factories supplying distribution centers, power plants supplying cities,
# and, as we will see, for assigning staff to tasks or students to rooms.

# %% [markdown]
# ### Modeling
#
# #### Problem Definition and Example
#
# A single good is shipped from $n$ sources to $m$ destinations. Source $i$ has a supply
# of $a_i$ units, destination $j$ has a demand of $b_j$ units, and shipping one unit over
# the link $i \to j$ costs $c_{ij}$. The problem is to find the cheapest way to ship the
# goods such that every demand is met and no supply is exceeded.
#
# As an example, two warehouses W1 and W2 supply three stores S1, S2 and S3, with the
# following transport costs per unit, supplies and demands:
#
# | source \ destination | S1 | S2 | S3 | supply $a_i$ |
# |---|---|---|---|---|
# | W1 | 4 | 6 | 8 | 30 |
# | W2 | 5 | 3 | 4 | 25 |
# | demand $b_j$ | 20 | 15 | 20 | |
#
# For example, shipping 10 units from W1 to S2 costs $10 \cdot 6 = 60$. Here the total
# supply, 55, equals the total demand, so every warehouse has to be emptied.
#
# #### Decision Variables
#
# What we have to decide is how much to ship over each link, so we define one decision
# variable per link:
#
# $$
# x_{ij} = \text{number of units shipped from source } i \text{ to destination } j,
# \quad i = 1, \dots, n,\ j = 1, \dots, m.
# $$
#
# A good check for a choice of decision variables is whether their values tell you
# everything you need: given all $x_{ij}$, we can compute the total cost and check
# whether the supplies and demands are respected, so these variables suffice.
#
# #### Objective
#
# The cost of link $i \to j$ is $c_{ij} x_{ij}$, so we minimize the total cost
#
# $$
# \sum_{i=1}^{n} \sum_{j=1}^{m} c_{ij} x_{ij}.
# $$
#
# If a link $i \to j$ does not exist, we can still use this formulation by giving that
# link a very large cost $c_{ij}$, so that the optimizer never uses it (or we simply leave
# the variable out, as the pulp code below does).
#
# #### Constraints
#
# The total amount shipped out of source $i$ cannot exceed its supply:
#
# $$
# \sum_{j=1}^{m} x_{ij} \le a_i, \quad i = 1, \dots, n.
# $$
#
# We use "$\le$" rather than "$=$" because a source does not have to ship everything it
# has. The total amount shipped into destination $j$ must cover its demand:
#
# $$
# \sum_{i=1}^{n} x_{ij} \ge b_j, \quad j = 1, \dots, m.
# $$
#
# Here "$=$" would also be correct, but it is not needed: shipping more than the demand
# only adds cost (assuming they are $>0$), so a minimizing optimizer never does it. Finally, we cannot ship
# negative amounts, so $x_{ij} \ge 0$. A feasible solution only exists if the total
# supply is at least the total demand, $\sum_i a_i \ge \sum_j b_j$.
#
# #### Complete LO Model
#
# $$
# \begin{aligned}
# \min \quad & \sum_{i=1}^{n} \sum_{j=1}^{m} c_{ij} x_{ij} \\
# \text{s.t.} \quad & \sum_{j=1}^{m} x_{ij} \le a_i, \quad i = 1, \dots, n \\
# & \sum_{i=1}^{n} x_{ij} \ge b_j, \quad j = 1, \dots, m \\
# & x_{ij} \ge 0, \quad i = 1, \dots, n,\ j = 1, \dots, m.
# \end{aligned}
# $$
#
# This is an LO model: we did not require the $x_{ij}$ to be integer. That is no loss
# here: when all supplies and demands are integer, the corner points of the feasible
# region of a transportation problem are integer, so the LO optimum found with the simplex method ships whole units
# anyway.

# %% [markdown]
# ### Modeling the Example
#
# Number the warehouses $i = 1, 2$ (W1, W2) and the stores $j = 1, 2, 3$ (S1, S2, S3).
# Filling in the numbers from the table gives
#
# $$
# \begin{aligned}
# \min \quad & 4x_{11} + 6x_{12} + 8x_{13} + 5x_{21} + 3x_{22} + 4x_{23} \\
# \text{s.t.} \quad & x_{11} + x_{12} + x_{13} \le 30 & \text{(supply W1)} \\
# & x_{21} + x_{22} + x_{23} \le 25 & \text{(supply W2)} \\
# & x_{11} + x_{21} \ge 20 & \text{(demand S1)} \\
# & x_{12} + x_{22} \ge 15 & \text{(demand S2)} \\
# & x_{13} + x_{23} \ge 20 & \text{(demand S3)} \\
# & x_{ij} \ge 0, \quad i = 1, 2,\ j = 1, 2, 3.
# \end{aligned}
# $$

# %% [markdown]
# ### Solving the Example in pulp
#
# We keep the data separate from the model, as in [Separating Data from the
# Model](lecture8_linear-optimization.ipynb). New here is that the decision variables
# are indexed by two things, a source and a destination, so we key the dictionary `ship`
# by `(source, destination)` tuples. The keys of `cost` also define which links exist:
# a variable is only created for a link that has a cost.

# %%
import pulp

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
print("status:", pulp.LpStatus[transport.status])
print("shipments:", {k: v.value() for k, v in ship.items() if v.value() > 0})
print("total cost:", transport.objective.value())

# %% [markdown]
# Each constraint gets a name (`supply_W1`, `demand_S3`, ...), which makes the printed
# model and any solver messages readable. The optimal plan ships whole units, as
# expected, and uses only four of the six links: W1 serves S1 and part of S2, and W2,
# which is cheap for S2 and S3, serves the rest.

# %% [markdown]
# ### Extensions
#
# (transshipment-problem)=
# #### Transshipment
#
# If goods can pass through intermediate nodes on the way from sources to destinations
# (for example, a cross-dock or a distribution center), we have the *transshipment
# problem*. The decision variables are again the amounts $x_{ij}$ on the links, now
# including links into and out of intermediate nodes. For every intermediate node $k$ we
# add a flow-conservation constraint: what comes in must go out:
#
# $$
# \sum_{i} x_{ik} = \sum_{j} x_{kj}.
# $$
#
# This extends to a full network by stacking several layers of intermediate nodes, and it
# is the model behind the [shortest path](lecture11_shortest-path.ipynb) and
# [maximum flow](lecture11_maximum-flow.ipynb) problems of Lecture 11.

# %% [markdown]
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.3 "Example LO Problems"
#   (transportation). Spreadsheet material replaced with pulp.
