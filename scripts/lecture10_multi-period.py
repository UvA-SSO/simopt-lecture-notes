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
# description: "Multi-period inventory planning as an LO model, with state variables that track the stock from one period to the next."
# thumbnail: null
# ---
# # Lecture 10: Multi-Period Inventory Planning
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture10_multi-period.ipynb)

# %% [markdown]
# This notebook covers multi-period inventory planning: deciding how much to order in
# each period so that all demand is met at the lowest total order and holding cost. It
# is our first multi-period model, in which a decision in one period affects all later
# periods, and it needs a new kind of variable, the state variable tracked over time. Like the applications of [Lecture 9](lecture9_introduction.ipynb), the problem
# is introduced in the same steps: a practical motivation, the generic model built up
# with the [four modeling steps](lecture8_linear-optimization.ipynb#modeling-approach),
# the model for a concrete example, its solution in pulp, and possible extensions.
#
# **Learning outcomes**
#
# On completion of this notebook, you will be able to:
#
# - formulate a multi-period inventory planning problem as an LO model;
# - use state variables and balance constraints to link consecutive periods;
# - extend the model with capacities, several products and fixed order costs.

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
# plus what is ordered for period $t$ (assuming it arrives at start of that period), minus what is sold during that period:
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
import pulp

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
# #### Capacities
#
# A storage capacity $S$ limits the stock, $s_t \le S$, and if the product is made rather
# than bought, a production capacity $P_t$ limits the amount made in period $t$,
# $x_t \le P_t$. Both are one extra constraint per period, without a new modeling trick.
#
# #### Several Products
#
# With several products, every variable and parameter gets a product index, and each
# product has its own balance constraints. Constraints for resources that the products
# share, such as storage space, link the products together.
#
# #### Fixed Order Costs
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
# ## References
#
# - Koole, G. (2019). *An Introduction to Business Analytics*. §6.3 (multi-period).
#   Spreadsheet material replaced with pulp.
