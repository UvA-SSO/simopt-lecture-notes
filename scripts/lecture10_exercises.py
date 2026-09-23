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
# # Lecture 10: Exercises
#
# [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/github/UvA-SSO/simopt-lecture-notes/blob/main/notebooks/lecture10_exercises.ipynb)

# %% [markdown]
# The smaller exercises embedded in
# [Modeling Tools and Solvers](lecture10_modeling-tools.ipynb) check what you just read.
# This notebook collects the larger exercises for Lecture 10: independent problems worth
# more time.
#
# :::{warning} Try It Yourself First
# The homework exercises below are representative of what you can expect on the exam:
# solve them by hand, pen-and-paper, without pulp or a computer. Attempt each one
# yourself, or make a serious effort, before opening the answer. If you do not manage to
# solve it, look at the answer to help you continue. Once solved, come back at a later
# time and try it again without looking at the answer. As extra practice, you can also
# solve them with pulp.
# :::

# %% [markdown]
# ## Homework Exercises

# %% [markdown]
# :::{exercise}
# :label: hw-10-1
#
# A supply chain specialist manages the inventory of a single product cost-efficiently
# over $T$ periods. Storing one unit at time $t$ costs $h_t > 0$; ordering one unit at
# time $t$ costs $c_t > 0$. Demand at time $t$ is $d_t$ and must always be met. Orders
# placed at time $t$ are available immediately at time $t$. The initial stock is $s_0$.
#
# Decision variables, for $t = 1, \dots, T$: $x_t$ = order size at time $t$, $s_t$ = stock
# level at time $t$. The ILO model is
#
# $$
# \min_{x_t, s_t} \sum_{t=1}^{T} (c_t x_t + h_t s_t) \quad \text{s.t.} \quad
# s_t = s_{t-1} - d_t + x_t,\ s_t, x_t \ge 0, \text{ for } t = 1, \dots, T.
# $$
#
# Solving it in pulp for $s_0 = 8$, $d = (5, 4, 8, 10, 4, 2, 1)$, $h_t = 1$ for all $t$,
# and $c = (10, 13, 13, 10, 10, 13, 13)$ (the code is given below) gives:
#
# ```
# Status: Optimal
# s_1 = 12.0   s_2 = 8.0   s_3 = 0.0   s_4 = 0.0   s_5 = 3.0   s_6 = 1.0   s_7 = 0.0
# x_1 = 9.0    x_2 = 0.0   x_3 = 0.0   x_4 = 10.0  x_5 = 7.0   x_6 = 0.0   x_7 = 0.0
# Total costs = 284.0
# ```
#
# ```python
# import pulp
#
# initial_stock = 8  # s_0
# demands = [None, 5, 4, 8, 10, 4, 2, 1]  # demands[t] = d_t
# holding_costs = [None, 1, 1, 1, 1, 1, 1, 1]  # holding_costs[t] = h_t
# order_costs = [None, 10, 13, 13, 10, 10, 13, 13]  # order_costs[t] = c_t
# max_periods = len(demands) - 1  # T
#
# ILO_problem = pulp.LpProblem("Inventory_problem", pulp.LpMinimize)
# supply = [None] + [
#     pulp.LpVariable(f"x_{t}", cat="Integer", lowBound=0) for t in range(1, max_periods + 1)
# ]
# stock = [initial_stock] + [
#     pulp.LpVariable(f"s_{t}", cat="Integer", lowBound=0) for t in range(1, max_periods + 1)
# ]
# ILO_problem += (
#     pulp.lpDot(order_costs[1:], supply[1:]) + pulp.lpDot(holding_costs[1:], stock[1:])
# ), "total_costs"
# for t in range(1, max_periods + 1):
#     ILO_problem += stock[t] == stock[t - 1] - demands[t] + supply[t], f"stock_balance_{t}"
#
# ILO_problem.solve(pulp.PULP_CBC_CMD(msg=False))
# print("Status:", pulp.LpStatus[ILO_problem.status])
# for v in ILO_problem.variables():
#     print(v.name, "=", v.varValue)
# ```
#
# Answer the following **without running the code**:
#
# a. Explain in words when new products should be ordered, and how many. Derive the total
#    holding cost and the total ordering cost.
#
# b. What is the maximum stock level over the time horizon?
#
# c. What happens if the specialist forgets the constraint $s_t, x_t \ge 0$ and solves the
#    model?
#
# d. What happens if the constraint $x_1 + x_2 = 0$ is added and the model is solved?
#    Explain it in words, and derive the answer using only the model's constraints.
#
# e. Explain how the pulp code needs to change to account for each of the following,
#    independently:
#
#    i. The maximum stock level cannot exceed 10.
#
#    ii. $T = 8$, with $d_8 = 8$, $c_8 = 2$, $h_8 = 3$ (the rest of the data unchanged).
#
#    iii. The holding costs in periods 4, 5, and 6 are 2 euro per unit.
#
#    iv. Any leftover stock at time $T$ costs 3 euro per unit as waste.
#
#    v. Any leftover stock at time $T$, regardless of amount, costs a fixed 30 euro.
#
#    vi. Orders must be in multiples of 5 (0, 5, 10, 15, ...).
#
#    vii. Orders can only be placed in periods 1, 3, and 5.
#
# f. The specialist is uncertain about demand and foresees 4 scenarios with probabilities
#    0.5, 0.15, 0.05, and 0.3, respectively (each a full demand vector over the 7
#    periods). All demand must still be met, in every scenario. Explain how to use the
#    model to achieve this at minimum cost.
#
# g. Three candidate warehouses are considered, each with its own maximum stock level
#    $C_i$ and rental cost $R_i$ for the whole horizon (replacing the per-unit holding
#    cost), with $C_1 < C_2 < C_3$ and $R_1 < R_2 < R_3$. Explain how to determine the
#    best warehouse using the model.
#
# h. Warehouse 1 from part g is rented, and a second product is now also stored there: the
#    combined stock of both products may not exceed the warehouse's capacity $R_1$, and
#    there are no holding costs. Explain how to change the model to find the
#    minimum-cost procurement strategy for both products (no code needed, just the
#    model).
# :::
#

# %% [markdown]
#
# :::{solution} hw-10-1
# :label: sol-hw-10-1
# :class: dropdown
#
# a. $x_t$ is the order size at time $t$, so from the output: order 9 units in period 1,
#    10 units in period 4, and 7 units in period 5 (all other $x_t = 0$). The holding
#    costs, from the $s_t$ values, are $12 + 8 + 0 + 0 + 3 + 1 + 0 = 24$ euro... more
#    precisely, using $h_t = 1$: $s_1 + s_2 + \dots + s_7 = 12 + 8 + 0 + 0 + 3 + 1 + 0 =
#    24$ euro. Since the total cost is 284 euro, the ordering cost is $284 - 24 = 260$
#    euro.
#
# b. The stock levels are $12, 8, 0, 0, 3, 1, 0$, so the maximum stock level is 12,
#    reached in period 1.
#
# c. Since $h_t > 0$ and $c_t > 0$ for all $t$, the optimizer wants $x_t$ and $s_t$ as
#    small as possible. Without the nonnegativity constraint, it would drive them to
#    $-\infty$, making the total cost unbounded below; pulp reports the model as
#    unbounded.
#
# d. The extra constraint forbids ordering in periods 1 and 2. With initial stock 8 and
#    demand $5 + 4 = 9$ over those two periods, demand cannot be met, so the model is
#    infeasible (pulp reports `Infeasible`). Using only the constraints: nonnegativity of
#    $s_2$ requires $s_1 - d_2 + x_2 \ge 0$; substituting $s_1 = s_0 - d_1 + x_1$ gives
#    $x_1 + x_2 \ge d_1 + d_2 - s_0 = 5 + 4 - 8 = 1$, so $x_1 + x_2 = 0$ is indeed
#    infeasible.
#
# e. i. Add `ILO_problem += stock[t] <= 10` for every $t$ (or pass `upBound=10` when
#       creating the stock variables).
#
#    ii. Extend the data lists: `demands = [None, 5, 4, 8, 10, 4, 2, 1, 8]`,
#        `holding_costs = [None, 1, 1, 1, 1, 1, 1, 1, 3]`,
#        `order_costs = [None, 10, 13, 13, 10, 10, 13, 13, 2]`.
#
#    iii. `holding_costs = [None, 1, 1, 1, 2, 2, 2, 1]`.
#
#    iv. Add `3 * stock[-1]` to the objective.
#
#    v. Introduce a binary `leftover_costs` variable, add `30 * leftover_costs` to the
#       objective, and enforce `stock[-1] <= leftover_costs * 100000` (a big-M
#       constraint) so it is forced to 1 whenever stock remains at time $T$.
#
#    vi. Introduce integer variables `order_multiple[t]` and add
#        `supply[t] == 5 * order_multiple[t]` for every $t$.
#
#    vii. Add `supply[t] == 0` for every $t \notin \{1, 3, 5\}$.
#
# f. Take, period by period, the worst case (maximum) demand across the four scenarios:
#    $d = (6, 5, 9, 12, 6, 3, 3)$, and re-optimize the model with this conservative demand
#    vector. This guarantees every scenario's demand is met.
#
# g. Remove the holding-cost term from the objective, then solve the model once per
#    warehouse (using $C_i$ as the stock upper bound) and add the corresponding $R_i$ to
#    each optimal cost. The warehouse with the lowest total cost is best, and its
#    solution gives the ordering strategy to use.
#
# h. Introduce a second set of parameters and variables for the new product, marked with a
#    prime: $x_t'$, $s_t'$, $c_t'$, $d_t'$, $s_0'$. The model becomes
#
#    $$
#    \begin{aligned}
#    \min_{x_t, s_t, x_t', s_t'} \quad & \sum_{t=1}^{T} (c_t x_t + c_t' x_t') \\
#    \text{s.t.} \quad & s_t = s_{t-1} - d_t + x_t \text{ for all } t \\
#    & s_t' = s_{t-1}' - d_t' + x_t' \text{ for all } t \\
#    & s_t + s_t' \le R_1 \text{ for all } t \\
#    & s_t, x_t, s_t', x_t' \ge 0 \text{ for all } t.
#    \end{aligned}
#    $$
#
#    This is the same model as before (minus holding costs) duplicated for the second
#    product, with the two products' stock levels linked through the shared capacity
#    constraint $s_t + s_t' \le R_1$.
# :::

# %% [markdown]
# :::{exercise}
# :label: hw-10-2
#
# a. Give three advantages of using pulp for solving ILO problems instead of Excel's
#    solver.
#
# b. Explain what an algebraic modeling language (AML) is used for. Name two examples.
#
# c. Rank the following ILO solvers from slow to fast: Gurobi, SCIP, Excel solver, CBC
#    solver. What is the reason not everyone uses the fastest one?
# :::
#

# %% [markdown]
#
# :::{solution} hw-10-2
# :label: sol-hw-10-2
# :class: dropdown
#
# a. pulp comes with a better solver than Excel's; pulp is open-source; and with pulp, the
#    data and the model are kept separate, so new instances (different sizes) can be
#    solved much more easily than by rebuilding an Excel sheet.
#
# b. An algebraic modeling language (AML), such as AMPL or pulp, lets you write an ILO
#    model in abstract terms; it then acts as the interface between data and a solver by
#    turning the abstract model into a concrete instance from the data, calling the
#    solver, and presenting the results. An AML is not itself a solver.
#
# c. From slow to fast: Excel solver, CBC, SCIP, Gurobi. The fastest is proprietary
#    (though a student license is available), which limits its use in practice.
# :::
